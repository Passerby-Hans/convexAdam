"""
0096 affine->ConvexAdam registration: warp pdff_seg into T2 space, assess seg
coverage quality (the user's deliverable: visually-correct, well-covering labels
on the original T2 image for downstream training).

Outputs (in this folder):
  warped_pdff_seg_aff.nii.gz   - affine+deformable warped pdff_seg (in T2 frame)
  warped_pdff_seg_raw.nii.gz   - deformable-only warped pdff_seg (for comparison)
  warped_pdff_seg_aff_onT2...  - also resampled back onto the ORIGINAL T2 grid
  coverage_metrics.txt         - per-organ Dice / precision / recall
  coverage_all_slices.png      - montage: T2 + true seg | T2 + warped seg | overlay (green=true, red=warped, yellow=agree)
  coverage_raw_vs_aff.png      - posterior-half overlay: RAW vs AFFINE-deformable

Run (convexadam venv, cwd = ConvexAdam root):
    PYTHONPATH=experiments python experiments/reg0096/affine_deform_0096.py
"""
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import nibabel as nib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # ConvexAdam/experiments (for imaging_utils)
from imaging_utils import nib_to_sitk, sitk_to_nib, resample_iso, crop_or_pad, warp_ras_with_disp  # noqa: E402
from convexAdam.convex_adam_MIND import convex_adam_pt  # noqa: E402

RAW = HERE.parent.parent.parent / "our-data" / "pdff_96-115"
SEG = HERE.parent.parent.parent / "our-data" / "BLA_0096_1"
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


def resample_to_fixed(fixed_sitk, moving_sitk, nn=False, default=0.0):
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(fixed_sitk)
    r.SetInterpolator(sitk.sitkNearestNeighbor if nn else sitk.sitkLinear)
    r.SetDefaultPixelValue(default)
    r.SetTransform(sitk.Transform())
    return r.Execute(moving_sitk)


def affine_align(fixed_sitk, moving_sitk):
    f = sitk.Cast(fixed_sitk, sitk.sitkFloat32)
    m = sitk.Cast(moving_sitk, sitk.sitkFloat32)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(32)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.15)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsGradientDescent(1.0, 300, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(sitk.CenteredTransformInitializer(f, m, sitk.AffineTransform(3)), inPlace=False)
    tx = R.Execute(f, m)
    print(f"  affine final MI metric = {R.GetMetricValue():.4f}")
    return sitk.Resample(m, f, tx, sitk.sitkLinear, 0.0), tx


def reg(t2_img, t2_aff, pdff_img, pdff_aff, use_affine, t2s_raw, t2s_aff, ps_raw, ps_aff):
    fixed = nib_to_sitk(t2_img, t2_aff)
    moving_raw = nib_to_sitk(pdff_img, pdff_aff)
    if use_affine:
        moving, tx = affine_align(fixed, moving_raw)
        ps_in = sitk.Resample(nib_to_sitk(ps_raw.astype(np.float32), ps_aff), fixed, tx, sitk.sitkNearestNeighbor, 0.0)
        fov_in = sitk.Resample(nib_to_sitk(np.ones_like(pdff_img, dtype=np.float32), pdff_aff), fixed, tx, sitk.sitkNearestNeighbor, 0.0)
    else:
        moving = resample_to_fixed(fixed, moving_raw)
        tx = None
        ps_in = resample_to_fixed(fixed, nib_to_sitk(ps_raw.astype(np.float32), ps_aff), nn=True)
        fov_in = resample_to_fixed(fixed, nib_to_sitk(np.ones_like(pdff_img, dtype=np.float32), pdff_aff), nn=True)
    disp = convex_adam_pt(fixed, moving)
    ps, _ = sitk_to_nib(ps_in)
    fov, _ = sitk_to_nib(fov_in)
    mov_arr, _ = sitk_to_nib(moving)
    warped_mov = warp_ras_with_disp(mov_arr, disp, order=1)   # warped PDFF image in T2 frame
    overlap = warp_ras_with_disp(fov, disp, order=0) > 0.5
    wseg = warp_ras_with_disp(ps, disp, order=0).astype(np.uint8)
    t2s, _ = sitk_to_nib(resample_to_fixed(fixed, nib_to_sitk(t2s_raw, t2s_aff), nn=True))
    fixed_r, faff = sitk_to_nib(fixed)
    return fixed_r, faff, t2s.astype(np.uint8), wseg, overlap, warped_mov


def metrics(t2s, wseg, overlap):
    labs = sorted(set(np.unique(t2s).tolist()) | set(np.unique(wseg).tolist()))
    labs = [int(v) for v in labs if v != 0]
    rows = []
    for v in labs:
        M = overlap
        A = (wseg == v) & M
        B = (t2s == v) & M
        inter = (A & B).sum()
        prec = inter / A.sum() if A.sum() else float("nan")
        rec = inter / B.sum() if B.sum() else float("nan")
        dice = 2 * inter / (A.sum() + B.sum()) if (A.sum() + B.sum()) else float("nan")
        rows.append((v, int(B.sum()), dice, prec, rec))
    return rows


def main():
    np.random.seed(0); torch.manual_seed(0)
    t2s_raw, t2s_aff = load_ras(SEG / "T2" / "abdomen_ax_T2_seg.nii.gz")
    ps_raw, ps_aff = load_ras(SEG / "pdff_seg.nii.gz")
    t2_img, t2_aff = pp_img(RAW / "T2" / "BLA_0096_1" / "abdomen_ax_T2.nii.gz")
    pdff_img, pdff_aff = pp_img(RAW / "pdff" / "BLA_0096_1.nii.gz")

    print("=== RAW (deformable only) ===")
    fr, faff, t2s, wseg_raw, ov, _ = reg(t2_img, t2_aff, pdff_img, pdff_aff, False, t2s_raw, t2s_aff, ps_raw, ps_aff)
    mraw = metrics(t2s, wseg_raw, ov)
    print("=== AFFINE -> ConvexAdam ===")
    fr, faff, t2s, wseg_aff, ov, warped_mov = reg(t2_img, t2_aff, pdff_img, pdff_aff, True, t2s_raw, t2s_aff, ps_raw, ps_aff)
    maff = metrics(t2s, wseg_aff, ov)

    # save warped segs (in T2 iso+crop frame)
    nib.save(nib.Nifti1Image(wseg_raw.astype(np.int16), faff), HERE / "warped_pdff_seg_raw.nii.gz")
    nib.save(nib.Nifti1Image(wseg_aff.astype(np.int16), faff), HERE / "warped_pdff_seg_aff.nii.gz")
    nib.save(nib.Nifti1Image(t2s.astype(np.int16), faff), HERE / "t2_seg_in_frame.nii.gz")
    nib.save(nib.Nifti1Image(fr, faff), HERE / "t2_in_frame.nii.gz")

    # metrics table
    with open(HERE / "coverage_metrics.txt", "w") as f:
        f.write("label  T2vx   | RAW Dice/Prec/Rec    | AFF Dice/Prec/Rec\n")
        for (v, n, d, p, r), (_, _, d2, p2, r2) in zip(mraw, maff):
            f.write(f"{v:>4} {n:>7} | {d:.3f}/{p:.3f}/{r:.3f} | {d2:.3f}/{p2:.3f}/{r2:.3f}\n")
        f.write(f"\nMEAN RAW Dice={np.nanmean([x[2] for x in mraw]):.3f}  AFF Dice={np.nanmean([x[2] for x in maff]):.3f}\n")
    print("per-organ (label, T2vx): RAW Dice/Prec/Rec | AFF Dice/Prec/Rec")
    for (v, n, d, p, r), (_, _, d2, p2, r2) in zip(mraw, maff):
        print(f"  {v:>3} ({n:>6}): RAW {d:.2f}/{p:.2f}/{r:.2f} | AFF {d2:.2f}/{p2:.2f}/{r2:.2f}")
    print(f"MEAN Dice RAW={np.nanmean([x[2] for x in mraw]):.3f}  AFF={np.nanmean([x[2] for x in maff]):.3f}")

    # coverage montage: ~28 subsampled slices, 3 panels each:
    #   [T2 + true seg] | [warped PDFF + warped seg] | [overlay G=true R=warped]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from imaging_utils import _remap_labels, _label_cmap

    labs = sorted(set(np.unique(t2s).tolist()) | set(np.unique(wseg_aff).tolist()))
    labs = [int(v) for v in labs if v != 0]
    cmap = _label_cmap(len(labs))
    vmax = len(labs)

    def nm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    Z = t2s.shape[2]
    step = max(1, Z // 28)
    zs = list(range(0, Z, step))[:28]
    cols = 7
    rowsn = int(np.ceil(len(zs) / cols))
    fig, axs = plt.subplots(rowsn, 3 * cols, figsize=(3 * cols * 1.5, rowsn * 1.6))
    axs = np.atleast_1d(axs).reshape(rowsn, 3 * cols)
    for i, z in enumerate(zs):
        r, c = divmod(i, cols)
        c0 = 3 * c
        img_t = nm(fr[:, :, z])
        img_w = nm(warped_mov[:, :, z])
        ts = _remap_labels(t2s[:, :, z], labs)
        ws = _remap_labels(wseg_aff[:, :, z], labs)
        # p0: T2 + true seg
        ax = axs[r][c0]
        ax.imshow(img_t.T, cmap="gray", origin="lower", aspect="equal")
        ax.imshow(ts.T, cmap=cmap, vmin=0, vmax=vmax, origin="lower", aspect="equal", interpolation="nearest")
        # p1: warped PDFF + warped seg
        ax = axs[r][c0 + 1]
        ax.imshow(img_w.T, cmap="gray", origin="lower", aspect="equal")
        ax.imshow(ws.T, cmap=cmap, vmin=0, vmax=vmax, origin="lower", aspect="equal", interpolation="nearest")
        # p2: overlay (G=T2 true, R=warped)
        ax = axs[r][c0 + 2]
        ax.imshow(img_t.T, cmap="gray", origin="lower", aspect="equal")
        ov = np.zeros((*ts.shape, 3))
        ov[..., 1] = (t2s[:, :, z] > 0).astype(float)
        ov[..., 0] = (wseg_aff[:, :, z] > 0).astype(float)
        ax.imshow(np.transpose(ov, (1, 0, 2)), origin="lower", aspect="equal")
        if i == 0:
            axs[r][c0].set_title("T2 + true seg", fontsize=6)
            axs[r][c0 + 1].set_title("warped PDFF + warped seg", fontsize=6)
            axs[r][c0 + 2].set_title("G=true R=warped", fontsize=6)
        for k in range(3):
            axs[r][c0 + k].set_xticks([])
            axs[r][c0 + k].set_yticks([])
    for i in range(len(zs), rowsn * cols):
        r, c = divmod(i, cols)
        for k in range(3):
            axs[r][3 * c + k].axis("off")
    fig.suptitle("0096 affine->ConvexAdam: T2 + true seg | warped PDFF + warped seg | "
                 "overlay (G=true R=warped yellow=agree)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "coverage_all_slices.png", dpi=80, bbox_inches="tight")
    print("saved coverage_all_slices.png (with warped PDFF) + warped segs + metrics")


if __name__ == "__main__":
    main()
