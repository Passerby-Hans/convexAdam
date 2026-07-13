"""
Validate ConvexAdam on OUR severely-mismatched abdominal data (BLA_0096_1).

Differences vs CHAOS (why this is a separate script, not the CHAOS code):
  * Data is already NIfTI (no DICOM) and mixes orientations (T1map/T2 are LAS,
    PDFF/masks are RAS) -> we nib.as_closest_canonical() everything to RAS on load.
  * Grids differ wildly: T1map 512x512x10 @20mm; T2 512x512x28 @7.5mm(6.5+1gap);
    PDFF 256x256x32 @8mm. ConvexAdam's resample-to-fixed handles the grid alignment.
  * 12-label abdominal segmentations exist in BOTH PDFF space (pdff_seg) and T2
    space (T2/abdomen_ax_T2_seg) -> we can score a real per-label Dice.

Experiment: register PDFF -> T2, warp pdff_seg into T2 space, per-label Dice vs
T2_seg on the FOV-overlap region. (PDFF/T2 slice thickness 8 vs 7.5 mm is the
comparable pair; T1map @20mm is the extreme case noted but has no seg to score.)

Run (WSL, convexadam venv, cwd = ConvexAdam root):
    PYTHONPATH=experiments python experiments/validate_ourdata.py
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
    warp_ras_with_disp, dice_label,
    save_montage, save_warped_montage, save_label_png,
)
from convexAdam.convex_adam_MIND import convex_adam_pt  # noqa: E402

DATA = Path(__file__).resolve().parent.parent.parent / "our-data" / "BLA_0096_1"
RUNS = Path(__file__).resolve().parent / "runs"
RUNS.mkdir(exist_ok=True)

ISO = (2.0, 2.0, 2.0)
INPLANE = 176  # in-plane crop; keep full z


def load_ras(name):
    im = nib.as_closest_canonical(nib.load(str(DATA / name)))  # LAS/RAS -> RAS
    return np.asarray(im.dataobj, np.float32), im.affine


def pp_img(name):
    img, aff = load_ras(name)
    img, aff = resample_iso(img, aff, ISO, order=1)
    img, aff = crop_or_pad(img, (INPLANE, INPLANE, img.shape[2]), aff)
    return img, aff


def pp_seg(name):
    lab, aff = load_ras(name)
    lab, _ = resample_iso(lab, aff, ISO, order=0)
    lab, _ = crop_or_pad(lab, (INPLANE, INPLANE, lab.shape[2]), aff, pad_value=0)
    return lab.astype(np.uint8)


def resample_to_fixed(fixed_sitk, moving_sitk, nn=False):
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(fixed_sitk)
    r.SetInterpolator(sitk.sitkNearestNeighbor if nn else sitk.sitkLinear)
    r.SetDefaultPixelValue(0.0)
    r.SetTransform(sitk.Transform())
    return r.Execute(moving_sitk)


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    # fixed = T2 (with T2_seg); moving = PDFF (with pdff_seg)
    t2_img, t2_aff = pp_img("abdomen_ax_T2.nii.gz")
    t2_seg = pp_seg("T2/abdomen_ax_T2_seg.nii.gz")
    pdff_img, pdff_aff = pp_img("pdff.nii.gz")
    pdff_seg = pp_seg("pdff_seg.nii.gz")
    print(f"T2  {t2_img.shape} (fixed)   PDFF {pdff_img.shape} (moving)")

    fixed_sitk = nib_to_sitk(t2_img, t2_aff)
    moving_sitk = resample_to_fixed(fixed_sitk, nib_to_sitk(pdff_img, pdff_aff))

    t0 = time.time()
    disp = convex_adam_pt(fixed_sitk, moving_sitk)
    print(f"ConvexAdam PDFF->T2: {time.time() - t0:.1f}s")

    # FOV overlap (PDFF coverage warped into T2 frame)
    fov_in = resample_to_fixed(fixed_sitk,
                               nib_to_sitk(np.ones_like(pdff_img, dtype=np.float32), pdff_aff), nn=True)
    fov_arr, _ = sitk_to_nib(fov_in)
    overlap = warp_ras_with_disp(fov_arr, disp, order=0) > 0.5

    # warp pdff_seg -> T2 frame (NN), and the PDFF image (linear) for viz
    seg_in = resample_to_fixed(fixed_sitk, nib_to_sitk(pdff_seg.astype(np.float32), pdff_aff), nn=True)
    seg_arr, _ = sitk_to_nib(seg_in)
    warped_seg = warp_ras_with_disp(seg_arr, disp, order=0).astype(np.uint8)
    mov_r, _ = sitk_to_nib(moving_sitk)
    warped_img = warp_ras_with_disp(mov_r, disp, order=1)
    t2_img_r, _ = sitk_to_nib(fixed_sitk)

    # labels present in either seg
    labs = sorted(set(np.unique(t2_seg).tolist()) | set(np.unique(pdff_seg).tolist()))
    labs = [int(v) for v in labs if v != 0]
    print(f"labels in segs: {labs}")

    print(f"\nFOV overlap: {int(overlap.sum())} vx ({100 * overlap.mean():.1f}% of T2 frame)")
    print("per-label Dice (warped pdff_seg vs T2_seg, on FOV overlap):")
    dice_tbl = {}
    for v in labs:
        d = dice_label(warped_seg, t2_seg, v, mask=overlap)
        dice_tbl[v] = d
        print(f"  label {v:2d}: Dice={d:.3f}   (T2 {int((t2_seg==v).sum())} vx | warped {int((warped_seg==v).sum())} vx)")
    valid = [dice_tbl[v] for v in labs if not np.isnan(dice_tbl[v])]
    print(f"\nmean Dice (over labels present in overlap): {np.mean(valid):.3f}")

    names = [f"L{v}" for v in labs]
    save_label_png(t2_img_r, t2_seg, warped_seg, overlap,
                   RUNS / "ourdata_pdff_to_t2.png", dice_tbl,
                   f"PDFF -> T2 label transfer (BLA_0096_1)  mean Dice={np.mean(valid):.3f}",
                   labels=labs, names=names)
    save_warped_montage(t2_img_r, t2_seg, warped_img, warped_seg, overlap,
                        RUNS / "ourdata_montage_pdff_to_t2.png",
                        f"PDFF->T2 warped vs T2 per slice (BLA_0096_1)", labels=labs)
    # per-modality raw montages (reoriented to RAS)
    t2r_img, _ = load_ras("abdomen_ax_T2.nii.gz")
    t2r_seg = np.asarray(nib.as_closest_canonical(nib.load(str(DATA / "T2/abdomen_ax_T2_seg.nii.gz"))).dataobj)
    pdffr_img, _ = load_ras("pdff.nii.gz")
    pdffr_seg = np.asarray(nib.as_closest_canonical(nib.load(str(DATA / "pdff_seg.nii.gz"))).dataobj)
    save_montage(t2r_img, t2r_seg, RUNS / "ourdata_montage_T2.png",
                 "T2 (BLA_0096_1) all axial slices + seg", labels=labs)
    save_montage(pdffr_img, pdffr_seg, RUNS / "ourdata_montage_PDFF.png",
                 "PDFF (BLA_0096_1) all axial slices + seg", labels=labs)
    print("-> ourdata_pdff_to_t2.png / ourdata_montage_pdff_to_t2.png / ourdata_montage_T2.png / ourdata_montage_PDFF.png")


if __name__ == "__main__":
    main()
