"""
Batch PDFF<->T2 registration on the new 20-case our-data dataset (BLA_0096..0115).

Each case: T2 (512,512,28 @0.742,0.742,7.5 LAS) + PDFF (256,256,32 @1.719,1.719,8 RAS).
Register PDFF -> T2 with ConvexAdam (MIND, modality-agnostic), report NCC before/after.
For BLA_0096_1 (the only case with segmentations, in ../our-data/BLA_0096_1/), also
compute per-label Dice (warped pdff_seg vs T2_seg) on the FOV overlap.

Run (convexadam venv, cwd = ConvexAdam root):
    PYTHONPATH=experiments python experiments/batch_register_ourdata.py
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
    warp_ras_with_disp, dice_label, ncc,
)
from convexAdam.convex_adam_MIND import convex_adam_pt  # noqa: E402

DATA = Path(__file__).resolve().parent.parent.parent / "our-data" / "pdff_96-115"
SEG_CASE = Path(__file__).resolve().parent.parent.parent / "our-data" / "BLA_0096_1"
RUNS = Path(__file__).resolve().parent / "runs"
RUNS.mkdir(exist_ok=True)

ISO = (2.0, 2.0, 2.0)
INPLANE = 176


def load_ras(p):
    im = nib.as_closest_canonical(nib.load(str(p)))
    return np.asarray(im.dataobj, np.float32), im.affine


def pp_img(p):
    arr, aff = load_ras(p)
    arr, aff = resample_iso(arr, aff, ISO, order=1)
    arr, aff = crop_or_pad(arr, (INPLANE, INPLANE, arr.shape[2]), aff)
    return arr, aff


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
    cases = sorted(p.name for p in (DATA / "pdff").glob("BLA_*.nii.gz"))
    print(f"{len(cases)} cases: {cases[0]} .. {cases[-1]}")

    # segs for 0096 (if present)
    seg_t2 = SEG_CASE / "T2" / "abdomen_ax_T2_seg.nii.gz"
    seg_pdff = SEG_CASE / "pdff_seg.nii.gz"
    have_seg0096 = seg_t2.exists() and seg_pdff.exists()

    rows = []
    for c in cases:
        cid = c.replace(".nii.gz", "")
        t2_img, t2_aff = pp_img(DATA / "T2" / cid / "abdomen_ax_T2.nii.gz")
        pdff_img, pdff_aff = pp_img(DATA / "pdff" / c)
        fixed_sitk = nib_to_sitk(t2_img, t2_aff)
        moving_sitk = resample_to_fixed(fixed_sitk, nib_to_sitk(pdff_img, pdff_aff))
        t0 = time.time()
        disp = convex_adam_pt(fixed_sitk, moving_sitk)
        secs = time.time() - t0
        mov_r, _ = sitk_to_nib(moving_sitk)
        warped = warp_ras_with_disp(mov_r, disp, order=1)
        fixed_r, _ = sitk_to_nib(fixed_sitk)
        before, after = ncc(fixed_r, mov_r), ncc(fixed_r, warped)

        dice_str = ""
        if cid == "BLA_0096_1" and have_seg0096:
            # resample both segs into the fixed (T2) grid (NN), warp pdff_seg by disp
            t2s_raw, t2s_aff = load_ras(seg_t2)
            ps_raw, ps_aff = load_ras(seg_pdff)
            t2s_arr, _ = sitk_to_nib(resample_to_fixed(fixed_sitk, nib_to_sitk(t2s_raw, t2s_aff), nn=True))
            ps_arr, _ = sitk_to_nib(resample_to_fixed(fixed_sitk, nib_to_sitk(ps_raw, ps_aff), nn=True))
            fov_arr, _ = sitk_to_nib(resample_to_fixed(
                fixed_sitk, nib_to_sitk(np.ones_like(pdff_img, dtype=np.float32), pdff_aff), nn=True))
            overlap = warp_ras_with_disp(fov_arr, disp, order=0) > 0.5
            wseg = warp_ras_with_disp(ps_arr, disp, order=0).astype(np.uint8)
            t2s_u = t2s_arr.astype(np.uint8)
            labs = sorted(set(np.unique(t2s_u).tolist()) | set(np.unique(wseg).tolist()))
            labs = [int(v) for v in labs if v != 0]
            ds = [dice_label(wseg, t2s_u, v, mask=overlap) for v in labs]
            valid = [d for d in ds if not np.isnan(d)]
            mean_d = np.mean(valid) if valid else float("nan")
            dice_str = f"  meanDice={mean_d:.3f} ({len(valid)} labels)"

        rows.append((c, before, after, secs))
        print(f"{c}: NCC {before:.3f}->{after:.3f}  [{secs:.1f}s]{dice_str}")

    print("\n=== summary ===")
    bf = np.mean([r[1] for r in rows]); af = np.mean([r[2] for r in rows])
    print(f"mean NCC before={bf:.3f} after={af:.3f}  (cross-modal, weak metric)")
    print(f"total reg time {sum(r[3] for r in rows):.1f}s for {len(rows)} cases")


if __name__ == "__main__":
    main()
