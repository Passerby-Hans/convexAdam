"""
ConvexAdam validation on REAL CHAOS T1/T2 (case 1).

Exp 1 — known deformation recovery (within T2SPIR):
    Apply a clean, realistic phi (affine + elastic, backward warp) to T2 to make
    a moving image, register moving -> fixed with ConvexAdam, and measure how
    well alignment is restored (NCC/MSE before vs after).

Exp 2 — real cross-modal registration (the actual task):
    Register T1DUAL_InPhase -> T2SPIR (different slice thickness / contrast) and
    report alignment.

All volumes stay in nibabel RAS (axis0=LR, axis1=AP, axis2=SI=head-foot); viz is
anatomically correct by construction.

Run (WSL, convexadam venv, cwd = ConvexAdam root):
    python experiments/validate_chaos.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imaging_utils import (  # noqa: E402
    nib_to_sitk, sitk_to_nib, resample_iso, crop_or_pad,
    generate_phi, backward_warp, ncc, ncc_roi, mse, save_comparison_png,
)
from convexAdam.convex_adam_MIND import convex_adam_pt  # noqa: E402
from convexAdam.apply_convex import apply_convex  # noqa: E402
from convexAdam.convex_adam_utils import resample_img, resample_moving_to_fixed  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "chaos_data" / "train" / "1"
RUNS = Path(__file__).resolve().parent / "runs"
RUNS.mkdir(exist_ok=True)

ISO_SPACING = (2.0, 2.0, 2.0)
CROP_SHAPE = (160, 160, 110)


def load_preprocessed(name):
    import nibabel as nib
    img = nib.load(str(DATA / f"{name}.nii.gz"))
    arr, aff = np.asarray(img.dataobj, np.float32), img.affine
    arr, aff = resample_iso(arr, aff, ISO_SPACING)
    arr, aff = crop_or_pad(arr, CROP_SHAPE, aff)
    return arr, aff


def register(fixed_arr, fixed_aff, moving_arr, moving_aff):
    """Align moving into fixed's frame, run ConvexAdam, return RAS arrays."""
    fixed_sitk = resample_img(nib_to_sitk(fixed_arr, fixed_aff), spacing=ISO_SPACING)
    moving_sitk = resample_moving_to_fixed(fixed_sitk, nib_to_sitk(moving_arr, moving_aff))
    t0 = time.time()
    disp = convex_adam_pt(fixed_sitk, moving_sitk)
    secs = time.time() - t0
    warped = apply_convex(disp, moving_sitk).transpose(2, 1, 0)  # (Z,Y,X)->(X,Y,Z) RAS
    fixed_r, _ = sitk_to_nib(fixed_sitk)
    moving_r, _ = sitk_to_nib(moving_sitk)
    return fixed_r, moving_r, warped, secs, disp


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    def norm01(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    print("== load & preprocess (2mm iso, crop to %s) ==" % str(CROP_SHAPE))
    t2, t2a = load_preprocessed("T2SPIR")
    t1, t1a = load_preprocessed("T1DUAL_InPhase")
    print(f"  T2SPIR         pp shape={t2.shape}")
    print(f"  T1DUAL_InPhase pp shape={t1.shape}")

    # ---- Exp 1a: identity sanity (T2 vs T2 -> expect NCC ~ 1.0) ----
    print("\n== Exp 1a: identity sanity (T2SPIR vs itself) ==")
    # 1a.1 backward_warp with phi=0 must be a no-op (relative RMSE ~ 0)
    zero_phi = np.zeros((*t2.shape, 3), np.float32)
    _idw = backward_warp(t2, zero_phi)
    _rel = np.sqrt(((_idw - t2) ** 2).mean()) / (t2.max() - t2.min() + 1e-8)
    print(f"  backward_warp(phi=0) relative RMSE = {_rel:.2e} (expect ~0)")
    assert _rel < 1e-3, "backward_warp(phi=0) not identity!"
    f0, _, w0, secs0, _ = register(t2, t2a, t2, t2a)
    print(f"  identity NCC={ncc(f0, w0):.4f} (expect ~=1.0)   nMSE={mse(norm01(f0), norm01(w0)):.6f}   [{secs0:.1f}s]")

    # ---- Exp 1b: known deformation recovery on T2 ----
    print("\n== Exp 1b: known deformation recovery (T2SPIR) ==")
    phi = generate_phi(t2.shape, sigma=10.0, magnitude=1.0, rot_deg=2.0, translation_vx=1.0)
    moving1 = backward_warp(t2, phi)
    pn = np.linalg.norm(phi, axis=-1)
    print(f"  applied |phi|: mean={pn.mean():.2f} max={pn.max():.2f} voxels")
    f1, m1, w1, secs, disp1 = register(t2, t2a, moving1, t2a)
    rec = np.linalg.norm(disp1, axis=-1)
    print(f"  ConvexAdam [{secs:.1f}s]; recovered |disp|: mean={rec.mean():.2f} max={rec.max():.2f}")
    print(f"  BEFORE reg: NCC={ncc(f1, m1):.4f}  ROI-NCC={ncc_roi(f1, m1):.4f}  nMSE={mse(norm01(f1), norm01(m1)):.6f}")
    print(f"  AFTER  reg: NCC={ncc(f1, w1):.4f}  ROI-NCC={ncc_roi(f1, w1):.4f}  nMSE={mse(norm01(f1), norm01(w1)):.6f}")
    save_comparison_png(
        f1, m1, w1, RUNS / "exp1_deformation_recovery.png",
        f"Exp1b T2SPIR known-deformation recovery  |  "
        f"|phi|~{pn.mean():.1f}vx  BEFORE NCC={ncc(f1,m1):.3f} -> AFTER NCC={ncc(f1,w1):.3f}",
    )
    print(f"  -> {RUNS/'exp1_deformation_recovery.png'}")

    # ---- Exp 2: real cross-modal T1 -> T2 ----
    # NOTE: raw-intensity NCC is a WEAK cross-modal metric (T1/T2 contrast differ);
    # it is reported for completeness, not as the verdict. Structural alignment is
    # judged from the MIND-driven registration + the figure.
    print("\n== Exp 2: real cross-modal T1DUAL_InPhase -> T2SPIR ==")
    f2, m2, w2, secs2, _ = register(t2, t2a, t1, t1a)
    print(f"  ConvexAdam [{secs2:.1f}s]  (cross-modal: NCC is weak, see figure for structure)")
    print(f"  BEFORE reg: NCC={ncc(f2, m2):.4f}  nMSE={mse(norm01(f2), norm01(m2)):.6f}")
    print(f"  AFTER  reg: NCC={ncc(f2, w2):.4f}  nMSE={mse(norm01(f2), norm01(w2)):.6f}")
    save_comparison_png(
        f2, m2, w2, RUNS / "exp2_t1_to_t2.png",
        f"Exp2 T1DUAL_InPhase -> T2SPIR (cross-modal, MIND-driven)  |  "
        f"BEFORE NCC={ncc(f2,m2):.3f} -> AFTER NCC={ncc(f2,w2):.3f}",
    )
    print(f"  -> {RUNS/'exp2_t1_to_t2.png'}")

    print("\n== summary ==")
    print(f"  1a identity         : NCC={ncc(f0, w0):.3f}")
    print(f"  1b deformation recov: NCC {ncc(f1, m1):.3f} -> {ncc(f1, w1):.3f}")
    print(f"  2  T1->T2 cross-mod : NCC {ncc(f2, m2):.3f} -> {ncc(f2, w2):.3f} (weak metric)")


if __name__ == "__main__":
    main()
