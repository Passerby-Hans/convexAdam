"""
C — does through-plane (z) resolution help cross-modal registration?

T1map has NO segmentation, so T1map->T2 cannot be Dice-scored. We simulate the
severe mismatch on a SEGMENTED modality (PDFF) and test whether finer through-plane
z improves registration to T2:

  * FINE  moving: PDFF as acquired        (32 sl @ 8mm)  -> register to T2 -> Dice_fine
  * THICK moving: PDFF z-downsampled 2.5x (13 sl @ 20mm, mimicking T1map)
                                                -> register to T2 -> Dice_thick

If Dice_fine > Dice_thick  -> through-plane resolution matters -> SR (recovers z)
   is worth pursuing. If ~equal -> SR won't help registration regardless of method.
This isolates the z-resolution hypothesis WITHOUT the ArSSR brain-domain-gap confound
(i.e. tests the principle before testing ArSSR specifically).

Run (convexadam venv, cwd = ConvexAdam root):
    PYTHONPATH=experiments python experiments/validate_sr_helps_reg.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import nibabel as nib
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imaging_utils import (  # noqa: E402
    nib_to_sitk, sitk_to_nib, resample_iso, crop_or_pad,
    warp_ras_with_disp, dice_label, save_all_slices_compare,
)
from convexAdam.convex_adam_MIND import convex_adam_pt  # noqa: E402

DATA = Path(__file__).resolve().parent.parent.parent / "our-data" / "BLA_0096_1"
RUNS = Path(__file__).resolve().parent / "runs"
RUNS.mkdir(exist_ok=True)

ISO = (2.0, 2.0, 2.0)
INPLANE = 176
THICKEN = 2.5  # simulate ~20mm acquisition (8mm * 2.5)


def load_ras(name):
    im = nib.as_closest_canonical(nib.load(str(DATA / name)))
    return np.asarray(im.dataobj, np.float32), im.affine


def z_down(arr, affine, factor, order):
    """Downsample z (axis 2) by `factor` (factor>1 -> fewer, thicker slices)."""
    out = zoom(arr.astype(np.float32), (1, 1, 1.0 / factor), order=order, mode="nearest")
    aff = affine.copy()
    aff[:3, 2] = affine[:3, 2] * factor  # z spacing grows by factor
    return out, aff


def pp_img(arr, aff):
    arr, aff = resample_iso(arr, aff, ISO, order=1)
    arr, aff = crop_or_pad(arr, (INPLANE, INPLANE, arr.shape[2]), aff)
    return arr, aff


def pp_seg(arr, aff):
    arr, _ = resample_iso(arr, aff, ISO, order=0)
    arr, _ = crop_or_pad(arr, (INPLANE, INPLANE, arr.shape[2]), aff, pad_value=0)
    return arr.astype(np.uint8)


def resample_to_fixed(fixed_sitk, moving_sitk, nn=False):
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(fixed_sitk)
    r.SetInterpolator(sitk.sitkNearestNeighbor if nn else sitk.sitkLinear)
    r.SetDefaultPixelValue(0.0)
    r.SetTransform(sitk.Transform())
    return r.Execute(moving_sitk)


def register_seg(fixed_img, fixed_aff, fixed_seg, mov_img, mov_aff, mov_seg):
    fixed_sitk = nib_to_sitk(fixed_img, fixed_aff)
    moving_sitk = resample_to_fixed(fixed_sitk, nib_to_sitk(mov_img, mov_aff))
    t0 = time.time()
    disp = convex_adam_pt(fixed_sitk, moving_sitk)
    dt = time.time() - t0
    fov_in = resample_to_fixed(fixed_sitk,
                               nib_to_sitk(np.ones_like(mov_img, dtype=np.float32), mov_aff), nn=True)
    fov_arr, _ = sitk_to_nib(fov_in)
    overlap = warp_ras_with_disp(fov_arr, disp, order=0) > 0.5
    seg_in = resample_to_fixed(fixed_sitk, nib_to_sitk(mov_seg.astype(np.float32), mov_aff), nn=True)
    seg_arr, _ = sitk_to_nib(seg_in)
    warped_seg = warp_ras_with_disp(seg_arr, disp, order=0).astype(np.uint8)
    fixed_r, _ = sitk_to_nib(fixed_sitk)
    return fixed_r, fixed_seg, warped_seg, overlap, dt


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    # fixed = T2 (img + T2_seg)
    t2_img, t2_aff = pp_img(*load_ras("abdomen_ax_T2.nii.gz"))
    t2_seg = pp_seg(*load_ras("T2/abdomen_ax_T2_seg.nii.gz"))
    # moving = PDFF (img + pdff_seg), fine + a thickened copy
    pdff_img0, pdff_aff0 = load_ras("pdff.nii.gz")
    pdff_seg0, _ = load_ras("pdff_seg.nii.gz")
    fine_img, fine_aff = pp_img(pdff_img0, pdff_aff0)
    fine_seg = pp_seg(pdff_seg0, pdff_aff0)
    thick_img_raw, thick_aff_raw = z_down(pdff_img0, pdff_aff0, THICKEN, order=1)
    thick_seg_raw, _ = z_down(pdff_seg0, pdff_aff0, THICKEN, order=0)
    thick_img, thick_aff = pp_img(thick_img_raw, thick_aff_raw)
    thick_seg = pp_seg(thick_seg_raw, thick_aff_raw)
    print(f"PDFF fine z={fine_img.shape[2]}  PDFF thick z={thick_img.shape[2]} (sim ~{8*THICKEN:.0f}mm)")

    labs = sorted(set(np.unique(t2_seg).tolist()) | set(np.unique(fine_seg).tolist()))
    labs = [int(v) for v in labs if v != 0]

    def dice_all(warped_seg, overlap):
        tbl = {}
        for v in labs:
            tbl[v] = dice_label(warped_seg, t2_seg, v, mask=overlap)
        return tbl

    print("\n=== FINE PDFF -> T2 ===")
    _, _, w_fine, ov_fine, dt = register_seg(t2_img, t2_aff, t2_seg, fine_img, fine_aff, fine_seg)
    d_fine = dice_all(w_fine, ov_fine)
    m_fine = np.nanmean([d_fine[v] for v in labs if not np.isnan(d_fine[v])])
    print(f"  reg {dt:.1f}s  mean Dice={m_fine:.3f}  overlap={100*ov_fine.mean():.1f}%")

    print("\n=== THICK PDFF (~20mm) -> T2 ===")
    _, _, w_thick, ov_thick, dt = register_seg(t2_img, t2_aff, t2_seg, thick_img, thick_aff, thick_seg)
    d_thick = dice_all(w_thick, ov_thick)
    m_thick = np.nanmean([d_thick[v] for v in labs if not np.isnan(d_thick[v])])
    print(f"  reg {dt:.1f}s  mean Dice={m_thick:.3f}  overlap={100*ov_thick.mean():.1f}%")

    print("\n=== per-label Dice (FINE vs THICK) ===")
    print(f"{'label':>6} {'FINE':>7} {'THICK':>7} {'diff':>7}")
    for v in labs:
        f, t = d_fine[v], d_thick[v]
        if np.isnan(f) and np.isnan(t):
            continue
        print(f"{v:>6} {f:>7.3f} {t:>7.3f} {f-t:>+7.3f}")
    print(f"\nMEAN   {m_fine:>7.3f} {m_thick:>7.3f} {m_fine-m_thick:>+7.3f}")
    print(f"-> {'through-plane z RESOLUTION MATTERS (SR worth pursuing)' if m_fine - m_thick > 0.01 else 'z-resolution barely matters (SR wont help registration much)'}")

    save_all_slices_compare(w_fine, w_thick, RUNS / "sr_helps_reg_fine_vs_thick.png",
                            "FINE PDFF->T2 warped seg", "THICK PDFF->T2 warped seg",
                            "C: does fine z help registration? warped seg (all axial slices)")


if __name__ == "__main__":
    main()
