"""
Label-transfer validation of T1->T2 registration (the modality-AGNOSTIC metric).

Pipeline:
  1. Register T1DUAL_InPhase -> T2SPIR (image-driven, ConvexAdam/MIND).
  2. Warp T1 organ labels into T2 space with NEAREST interpolation (labels discrete).
  3. Restrict to the FOV-overlap region (where T1 actually has scanned data after
     being mapped into the T2 frame). Voxels scanned by only one modality excluded.
  4. Per-organ Dice (liver / spleen / R-kidney / L-kidney) + label-overlay figure.

Unlike raw-intensity NCC, organ labels carry the same meaning in T1 and T2, so
Dice is a fair cross-modal score.

Run (WSL, convexadam venv, cwd = ConvexAdam root):
    PYTHONPATH=experiments python experiments/validate_labels.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imaging_utils import (  # noqa: E402
    nib_to_sitk, sitk_to_nib, resample_iso, crop_or_pad,
    warp_ras_with_disp, dice_label, ORGANS, save_label_png,
    save_montage, save_warped_montage,
)
from convexAdam.convex_adam_MIND import convex_adam_pt  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "chaos_data" / "train" / "1"
RUNS = Path(__file__).resolve().parent / "runs"
RUNS.mkdir(exist_ok=True)

ISO = (2.0, 2.0, 2.0)
INPLANE = 176  # in-plane crop; z is kept full (T1/T2 have different slice counts)


def resample_to_fixed(fixed_sitk, moving_sitk, nn=False):
    """Resample moving into fixed's frame (identity transform, geometry-only).
    nn=True -> nearest neighbor (labels/FOV masks); default linear (images).
    Outside the moving's FOV -> 0."""
    res = sitk.ResampleImageFilter()
    res.SetReferenceImage(fixed_sitk)
    res.SetInterpolator(sitk.sitkNearestNeighbor if nn else sitk.sitkLinear)
    res.SetDefaultPixelValue(0.0)
    res.SetTransform(sitk.Transform())
    return res.Execute(moving_sitk)


def load_pp(name):
    """Load image + label (RAS), iso-resample (linear / NN), in-plane crop. Label
    shares the image affine (written so by convert_chaos.py)."""
    ni = nib.load(str(DATA / f"{name}.nii.gz"))
    nl = nib.load(str(DATA / f"{name}_label.nii.gz"))
    img0, aff0 = np.asarray(ni.dataobj, np.float32), ni.affine
    lab0 = np.asarray(nl.dataobj, np.float32)
    img, aff = resample_iso(img0, aff0, ISO, order=1)
    lab, _ = resample_iso(lab0, aff0, ISO, order=0)
    tgt = (INPLANE, INPLANE, img.shape[2])  # keep full z
    img, aff = crop_or_pad(img, tgt, aff)
    lab, _ = crop_or_pad(lab, tgt, aff, pad_value=0)
    return img, lab.astype(np.uint8), aff


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    t2_img, t2_lab, t2_aff = load_pp("T2SPIR")
    t1_img, t1_lab, t1_aff = load_pp("T1DUAL_InPhase")
    print(f"T2 img {t2_img.shape}  T1 img {t1_img.shape}")

    fixed_sitk = nib_to_sitk(t2_img, t2_aff)  # already 2mm iso

    # 1) image registration T1 -> T2
    moving_sitk = resample_to_fixed(fixed_sitk, nib_to_sitk(t1_img, t1_aff))
    t0 = time.time()
    disp = convex_adam_pt(fixed_sitk, moving_sitk)
    print(f"ConvexAdam registration: {time.time() - t0:.1f}s")

    # 2) FOV-overlap mask: T1's coverage resampled into T2 frame, then deformed
    fov_in = resample_to_fixed(fixed_sitk,
                               nib_to_sitk(np.ones_like(t1_img, dtype=np.float32), t1_aff), nn=True)
    fov_arr, _ = sitk_to_nib(fov_in)
    overlap = warp_ras_with_disp(fov_arr, disp, order=0) > 0.5

    # 3) warp T1 labels into T2 space (NN throughout)
    lab_in = resample_to_fixed(fixed_sitk, nib_to_sitk(t1_lab.astype(np.float32), t1_aff), nn=True)
    lab_arr, _ = sitk_to_nib(lab_in)
    warped_lab = warp_ras_with_disp(lab_arr, disp, order=0).astype(np.uint8)

    # fixed-frame arrays
    t2_img_r, _ = sitk_to_nib(fixed_sitk)
    moving_r, _ = sitk_to_nib(moving_sitk)
    warped_img = warp_ras_with_disp(moving_r, disp, order=1)  # warped T1 image (t2wrap)

    # 4) per-organ Dice on the FOV overlap
    print(f"\nFOV overlap: {int(overlap.sum())} voxels ({100 * overlap.mean():.1f}% of T2 frame)")
    print("per-organ Dice (warped-T1 label vs T2 ground-truth, on FOV overlap):")
    dice_tbl = {}
    for v, oname in ORGANS.items():
        d = dice_label(warped_lab, t2_lab, v, mask=overlap)
        dice_tbl[v] = d
        t2_has = int((t2_lab == v).sum())
        t1_has = int((warped_lab == v).sum())
        print(f"  {oname:10s} Dice={d:.3f}   (T2 {t2_has} vx | warped-T1 {t1_has} vx)")

    mean_dice = np.nanmean([dice_tbl[v] for v in ORGANS])
    print(f"\nmean Dice (organs present): {mean_dice:.3f}")

    save_label_png(
        t2_img_r, t2_lab, warped_lab, overlap,
        RUNS / "label_transfer_t1_to_t2.png",
        dice_tbl,
        f"T1DUAL_InPhase -> T2SPIR label transfer (case 1)  mean Dice={mean_dice:.3f}",
    )
    print(f"-> {RUNS / 'label_transfer_t1_to_t2.png'}")

    # full-volume montages: T1 original, T2 original, warped-T1 vs T2 per slice
    def raw(s):
        return (np.asarray(nib.load(str(DATA / f"{s}.nii.gz")).dataobj, np.float32),
                np.asarray(nib.load(str(DATA / f"{s}_label.nii.gz")).dataobj))
    t1r_img, t1r_lab = raw("T1DUAL_InPhase")
    t2r_img, t2r_lab = raw("T2SPIR")
    save_montage(t1r_img, t1r_lab, RUNS / "montage_T1DUAL_InPhase.png",
                 "T1DUAL_InPhase (case 1) - all axial slices + organ labels")
    save_montage(t2r_img, t2r_lab, RUNS / "montage_T2SPIR.png",
                 "T2SPIR (case 1) - all axial slices + organ labels")
    print(f"-> {RUNS / 'montage_T1DUAL_InPhase.png'}")
    print(f"-> {RUNS / 'montage_T2SPIR.png'}")
    save_warped_montage(
        t2_img_r, t2_lab, warped_img, warped_lab, overlap,
        RUNS / "montage_warped_t1_to_t2.png",
        f"T1->T2 warped vs T2, per organ slice (case 1)  mean Dice={mean_dice:.3f}",
    )
    print(f"-> {RUNS / 'montage_warped_t1_to_t2.png'}")


if __name__ == "__main__":
    main()
