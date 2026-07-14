"""
Test the user's hypothesis: thick-z -> posterior (later) half can't be registered
(MIND can't find correspondences across 7.5-8mm gaps); ECLARE SR (denser z WITH
learned structure, vs linear-interp which is smooth/uninformative) should recover
the posterior-half alignment.

Compare RAW registration vs ECLARE-SR registration:
  * Dice overall / anterior-half / posterior-half (z-split) on BLA_0096_1.
  * Visual: posterior-half axial slices, T2 img + true label vs warped label (raw vs SR).

Goal: visually-correct, correctly-bounded labels on the original images for downstream training.

Run (convexadam venv, cwd = ConvexAdam root):
    PYTHONPATH=experiments python experiments/test_sr_posterior.py
"""
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imaging_utils import (  # noqa: E402
    nib_to_sitk, sitk_to_nib, resample_iso, crop_or_pad,
    warp_ras_with_disp, dice_label, _remap_labels, _label_cmap,
)
from convexAdam.convex_adam_MIND import convex_adam_pt  # noqa: E402

RAW = Path(__file__).resolve().parent.parent.parent / "our-data" / "pdff_96-115"
ECL = Path(__file__).resolve().parent.parent.parent / "ECLARE" / "runs" / "0096"
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


def resample_to_fixed(fixed_sitk, moving_sitk, nn=False):
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(fixed_sitk)
    r.SetInterpolator(sitk.sitkNearestNeighbor if nn else sitk.sitkLinear)
    r.SetDefaultPixelValue(0.0)
    r.SetTransform(sitk.Transform())
    return r.Execute(moving_sitk)


def register(t2_img, t2_aff, pdff_img, pdff_aff, t2s_raw, t2s_aff, ps_raw, ps_aff):
    fixed = nib_to_sitk(t2_img, t2_aff)
    moving = resample_to_fixed(fixed, nib_to_sitk(pdff_img, pdff_aff))
    disp = convex_adam_pt(fixed, moving)
    t2s, _ = sitk_to_nib(resample_to_fixed(fixed, nib_to_sitk(t2s_raw, t2s_aff), nn=True))
    ps, _ = sitk_to_nib(resample_to_fixed(fixed, nib_to_sitk(ps_raw, ps_aff), nn=True))
    fov, _ = sitk_to_nib(resample_to_fixed(
        fixed, nib_to_sitk(np.ones_like(pdff_img, dtype=np.float32), pdff_aff), nn=True))
    overlap = warp_ras_with_disp(fov, disp, order=0) > 0.5
    wseg = warp_ras_with_disp(ps, disp, order=0).astype(np.uint8)
    t2s = t2s.astype(np.uint8)
    fixed_r, _ = sitk_to_nib(fixed)
    return fixed_r, t2s, wseg, overlap, disp


def dice_split(t2s, wseg, overlap):
    Z = t2s.shape[2]
    mid = Z // 2
    labs = sorted(set(np.unique(t2s).tolist()) | set(np.unique(wseg).tolist()))
    labs = [int(v) for v in labs if v != 0]

    def mean(mask):
        ds = [dice_label(wseg, t2s, v, mask=(mask & overlap)) for v in labs]
        ds = [d for d in ds if not np.isnan(d)]
        return np.mean(ds) if ds else float("nan")
    om = np.ones_like(overlap)
    ant = np.zeros_like(overlap)
    ant[:, :, :mid] = True
    post = np.zeros_like(overlap)
    post[:, :, mid:] = True
    return mean(om), mean(ant), mean(post), mid


def main():
    np.random.seed(0); torch.manual_seed(0)
    t2s_raw, t2s_aff = load_ras(SEG / "T2" / "abdomen_ax_T2_seg.nii.gz")
    ps_raw, ps_aff = load_ras(SEG / "pdff_seg.nii.gz")

    print("=== RAW registration ===")
    t2i, t2a = pp_img(RAW / "T2" / "BLA_0096_1" / "abdomen_ax_T2.nii.gz")
    pdi, pda = pp_img(RAW / "pdff" / "BLA_0096_1.nii.gz")
    fr, t2s, wseg, ov, _ = register(t2i, t2a, pdi, pda, t2s_raw, t2s_aff, ps_raw, ps_aff)
    raw_all, raw_ant, raw_post, mid = dice_split(t2s, wseg, ov)
    print(f"  Dice overall={raw_all:.3f}  anterior={raw_ant:.3f}  posterior={raw_post:.3f}  (z-split at {mid})")
    raw_t2s, raw_wseg, raw_fr = t2s, wseg, fr

    print("=== ECLARE-SR registration ===")
    t2i, t2a = pp_img(ECL / "T2" / "abdomen_ax_T2_eclare.nii.gz")
    pdi, pda = pp_img(ECL / "PDFF" / "BLA_0096_1_eclare.nii.gz")
    fr, t2s, wseg, ov, _ = register(t2i, t2a, pdi, pda, t2s_raw, t2s_aff, ps_raw, ps_aff)
    sr_all, sr_ant, sr_post, _ = dice_split(t2s, wseg, ov)
    print(f"  Dice overall={sr_all:.3f}  anterior={sr_ant:.3f}  posterior={sr_post:.3f}")

    print("\n=== posterior-half (the failing region) ===")
    print(f"  RAW  posterior Dice = {raw_post:.3f}")
    print(f"  SR   posterior Dice = {sr_post:.3f}   (delta = {sr_post - raw_post:+.3f})")
    print(f"  overall RAW {raw_all:.3f} -> SR {sr_all:.3f} (delta {sr_all-raw_all:+.3f})")
    verdict = "ECLARE-SR HELPS the posterior half (hypothesis supported)" if sr_post - raw_post > 0.01 else "SR does NOT clearly help posterior (hypothesis not supported)"
    print("  ->", verdict)

    # visual: posterior-half label alignment, RAW vs SR
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = sorted(set(np.unique(raw_t2s).tolist()) | set(np.unique(raw_wseg).tolist()))
    labs = [int(v) for v in labs if v != 0]
    cmap = _label_cmap(len(labs))
    vmax = len(labs)

    def nm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    # pick 6 evenly-spaced posterior-half slices
    Z = raw_t2s.shape[2]
    zs = list(range(mid, Z, max(1, (Z - mid) // 6)))[:6]
    fig, axs = plt.subplots(len(zs), 4, figsize=(17, 2.6 * len(zs)), squeeze=False)
    # 4-col layout per slice: T2img+T2true | RAW warped | T2img+T2true | SR warped
    for r, z in enumerate(zs):
        img = nm(raw_fr[:, :, z])
        ttrue = _remap_labels(raw_t2s[:, :, z], labs)
        wraw = _remap_labels(raw_wseg[:, :, z], labs)
        wsr = _remap_labels(wseg[:, :, z], labs)
        for c, (lab, ttl) in enumerate([
            (ttrue, f"z={z} T2 TRUE label"),
            (wraw, "RAW reg: warped PDFF label"),
            (ttrue, f"z={z} T2 TRUE (ref)"),
            (wsr, "SR reg: warped PDFF label")]):
            ax = axs[r][c]
            ax.imshow(img.T, cmap="gray", origin="lower", aspect="equal")
            ax.imshow(lab.T, cmap=cmap, vmin=0, vmax=vmax, origin="lower", aspect="equal", interpolation="nearest")
            ax.set_title(ttl, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Posterior-half label alignment: RAW vs ECLARE-SR registration "
                 f"(post Dice {raw_post:.2f}->{sr_post:.2f})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(RUNS / "sr_posterior_label_align.png", dpi=90, bbox_inches="tight")
    print(f"saved {RUNS/'sr_posterior_label_align.png'}")


if __name__ == "__main__":
    main()
