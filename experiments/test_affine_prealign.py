"""
Test affine pre-alignment before ConvexAdam (the missing global step).

Hypothesis: T2 and PDFF are acquired in DIFFERENT breath-holds -> large global
translation/rotation. ConvexAdam is deformable-only and can't recover a large
global offset (esp. at the posterior/diaphragm edge -> the failing later half).
The standard pipeline is affine-then-deformable (ANTs: Affine->SyN). ConvexAdam
alone skips affine.

Compare on BLA_0096_1:
  * RAW:   ConvexAdam only (current).
  * AFF:   SimpleITK affine (Mattes MI, multi-res) PDFF->T2, THEN ConvexAdam.
Dice overall / anterior / posterior (z-split). If AFF jumps the posterior,
global offset was the cause.

Run (convexadam venv, cwd = ConvexAdam root):
    PYTHONPATH=experiments python experiments/test_affine_prealign.py
"""
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imaging_utils import nib_to_sitk, sitk_to_nib, resample_iso, crop_or_pad, warp_ras_with_disp, dice_label  # noqa: E402
from convexAdam.convex_adam_MIND import convex_adam_pt  # noqa: E402

RAW = Path(__file__).resolve().parent.parent.parent / "our-data" / "pdff_96-115"
SEG = Path(__file__).resolve().parent.parent.parent / "our-data" / "BLA_0096_1"
RUNS = Path(__file__).resolve().parent / "runs"
RUNS.mkdir(exist_ok=True)
ISO = (1.5, 1.5, 1.5)
INPLANE = 200


def load_ras(p):
    im = nib.as_closest_canonical(nib.load(str(p)))
    return np.asarray(im.dataobj, np.float32), im.affine


def pp_img(p):
    arr, aff = load_ras(p)
    arr, aff = resample_iso(arr, aff, ISO, order=1)
    arr, aff = crop_or_pad(arr, (INPLANE, INPLANE, arr.shape[2]), aff)
    return arr, aff


def affine_align(fixed_sitk, moving_sitk):
    """Affine-register moving->fixed (Mattes MI, multi-res). Returns (moving resampled
    into fixed frame, transform tx with tx mapping fixed->moving for seg resampling)."""
    f = sitk.Cast(fixed_sitk, sitk.sitkFloat32)
    m = sitk.Cast(moving_sitk, sitk.sitkFloat32)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(32)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.15)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=300,
                                    convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    init = sitk.CenteredTransformInitializer(f, m, sitk.AffineTransform(3))
    R.SetInitialTransform(init, inPlace=False)
    tx = R.Execute(f, m)
    aligned = sitk.Resample(m, f, tx, sitk.sitkLinear, 0.0)  # moving into fixed frame
    print(f"  affine MI metric final = {R.GetMetricValue():.4f}")
    return aligned, tx


def resample_to_fixed(fixed_sitk, moving_sitk, nn=False):
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(fixed_sitk)
    r.SetInterpolator(sitk.sitkNearestNeighbor if nn else sitk.sitkLinear)
    r.SetDefaultPixelValue(0.0)
    r.SetTransform(sitk.Transform())
    return r.Execute(moving_sitk)


def dice_zsplit(t2s, wseg, overlap):
    Z = t2s.shape[2]; mid = Z // 2
    labs = sorted(set(np.unique(t2s).tolist()) | set(np.unique(wseg).tolist()))
    labs = [int(v) for v in labs if v != 0]
    def mean(mask):
        ds = [dice_label(wseg, t2s, v, mask=(mask & overlap)) for v in labs]
        ds = [d for d in ds if not np.isnan(d)]
        return np.mean(ds) if ds else float("nan")
    om = np.ones_like(overlap); ant = np.zeros_like(overlap); ant[:, :, :mid] = True
    post = np.zeros_like(overlap); post[:, :, mid:] = True
    return mean(om), mean(ant), mean(post), mid


def run_case(t2_img, t2_aff, pdff_img, pdff_aff, t2s_raw, t2s_aff, ps_raw, ps_aff, use_affine):
    fixed = nib_to_sitk(t2_img, t2_aff)
    moving_sitk_raw = nib_to_sitk(pdff_img, pdff_aff)
    if use_affine:
        moving_sitk, tx = affine_align(fixed, moving_sitk_raw)
    else:
        moving_sitk = resample_to_fixed(fixed, moving_sitk_raw)
        tx = None
    disp = convex_adam_pt(fixed, moving_sitk)
    t2s, _ = sitk_to_nib(resample_to_fixed(fixed, nib_to_sitk(t2s_raw, t2s_aff), nn=True))
    if tx is not None:
        ps_in_sitk = sitk.Resample(nib_to_sitk(ps_raw.astype(np.float32), ps_aff), fixed, tx,
                                    sitk.sitkNearestNeighbor, 0.0)
        fov_sitk = sitk.Resample(nib_to_sitk(np.ones_like(pdff_img, dtype=np.float32), ps_aff), fixed, tx,
                                  sitk.sitkNearestNeighbor, 0.0)
    else:
        ps_in_sitk = resample_to_fixed(fixed, nib_to_sitk(ps_raw.astype(np.float32), ps_aff), nn=True)
        fov_sitk = resample_to_fixed(fixed, nib_to_sitk(np.ones_like(pdff_img, dtype=np.float32), pdff_aff), nn=True)
    ps, _ = sitk_to_nib(ps_in_sitk)
    fov, _ = sitk_to_nib(fov_sitk)
    overlap = warp_ras_with_disp(fov, disp, order=0) > 0.5
    wseg = warp_ras_with_disp(ps, disp, order=0).astype(np.uint8)
    t2s = t2s.astype(np.uint8)
    return dice_zsplit(t2s, wseg, overlap)


def main():
    np.random.seed(0); torch.manual_seed(0)
    t2s_raw, t2s_aff = load_ras(SEG / "T2" / "abdomen_ax_T2_seg.nii.gz")
    ps_raw, ps_aff = load_ras(SEG / "pdff_seg.nii.gz")
    t2_img, t2_aff = pp_img(RAW / "T2" / "BLA_0096_1" / "abdomen_ax_T2.nii.gz")
    pdff_img, pdff_aff = pp_img(RAW / "pdff" / "BLA_0096_1.nii.gz")

    print("=== RAW (ConvexAdam only) ===")
    a_all, a_ant, a_post, _ = run_case(t2_img, t2_aff, pdff_img, pdff_aff,
                                       t2s_raw, t2s_aff, ps_raw, ps_aff, use_affine=False)
    print(f"  overall={a_all:.3f}  anterior={a_ant:.3f}  posterior={a_post:.3f}")

    print("=== AFFINE pre-align + ConvexAdam ===")
    b_all, b_ant, b_post, _ = run_case(t2_img, t2_aff, pdff_img, pdff_aff,
                                       t2s_raw, t2s_aff, ps_raw, ps_aff, use_affine=True)
    print(f"  overall={b_all:.3f}  anterior={b_ant:.3f}  posterior={b_post:.3f}")

    print("\n=== posterior-half (failing region) ===")
    print(f"  RAW     posterior = {a_post:.3f}")
    print(f"  AFFINE  posterior = {b_post:.3f}   (delta {b_post-a_post:+.3f})")
    print(f"  overall RAW {a_all:.3f} -> AFF {b_all:.3f} (delta {b_all-a_all:+.3f})")
    print("  ->", "AFFINE pre-align HELPS posterior (global offset was the cause)" if b_post-a_post > 0.02
          else "affine did NOT clearly help (cause is local anatomy/coverage)")


if __name__ == "__main__":
    main()
