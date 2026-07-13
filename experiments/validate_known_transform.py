"""
Validate ConvexAdam registration (bundled anisotropic test) with anatomically
correct, de-skewed visualization.

Why this rewrite: the test volumes are OBLIQUE (~23 deg in the y-z plane — see
their direction matrix). Slicing the raw array axes therefore produced tilted
("歪的") slices. Here we first resample every volume to an identity-direction
(axis-aligned) grid in the fixed image's physical frame, then reorder to a
canonical anatomical layout so that:

  numpy axis 0 ~ Left-Right   (physical x, LPS)
  numpy axis 1 ~ Anterior-Posterior (physical y)
  numpy axis 2 ~ Superior-Inferior (physical z)   <-- head-foot / 身高方向

Then: axial   = slice along axis 2 (SI / head-foot)
      coronal = slice along axis 1 (AP)
      sagittal= slice along axis 0 (LR)

Inputs: ConvexAdam/tests/output/10000/  (from test_convex_adam_mind_aniso.py)
Output: ConvexAdam/experiments/figures/validate_known_transform.png

Run (WSL, convexadam venv, cwd = ConvexAdam root):
    python experiments/validate_known_transform.py
"""
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUBJ = "10000_1000000"
OUT = Path(__file__).resolve().parent.parent / "tests" / "output" / "10000"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True, parents=True)


def axis_aligned_reference(img):
    """Build an identity-direction SimpleITK image covering the same physical
    bounding box as ``img`` (same spacing). De-skews oblique acquisitions."""
    spacing = np.array(img.GetSpacing(), float)
    D = np.array(img.GetDirection()).reshape(3, 3)
    size = np.array(img.GetSize(), int)
    origin = np.array(img.GetOrigin(), float)
    corners = []
    for i in (0, size[0] - 1):
        for j in (0, size[1] - 1):
            for k in (0, size[2] - 1):
                corners.append(origin + D @ (np.array([i, j, k]) * spacing))
    corners = np.array(corners)
    pmin, pmax = corners.min(axis=0), corners.max(axis=0)
    new_size = np.maximum(1, np.round((pmax - pmin) / spacing)).astype(int)
    ref = sitk.Image([int(s) for s in new_size], img.GetPixelID())
    ref.SetSpacing(tuple(spacing))
    ref.SetOrigin(tuple(pmin))
    ref.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return ref


def to_ras_array(path, ref):
    """Resample image into the shared axis-aligned ``ref`` grid and return a
    numpy array ordered (LR, AP, SI)."""
    img = sitk.Resample(sitk.ReadImage(str(path)), ref, sitk.Transform(), sitk.sitkLinear)
    arr = sitk.GetArrayFromImage(img)            # (z=SI, y=AP, x=LR)
    return np.transpose(arr, (2, 1, 0)).astype(np.float32)  # -> (LR, AP, SI)


fixed0 = sitk.ReadImage(str(OUT / f"{SUBJ}_fixed_resampled.mha"))
ref = axis_aligned_reference(fixed0)
fa = to_ras_array(OUT / f"{SUBJ}_fixed_resampled.mha", ref)
ma = to_ras_array(OUT / f"{SUBJ}_moving_rotated_and_shifted_resampled.mha", ref)
wa = to_ras_array(OUT / f"{SUBJ}_moving_rotated_and_shifted_resampled_warped.mha", ref)


def norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


fa, ma, wa = norm(fa), norm(ma), norm(wa)


def ncc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()) + 1e-12))


def mse(a, b):
    return float(((a - b) ** 2).mean())


print("=== alignment vs fixed (de-skewed, anatomical frame) ===")
print(f"moving BEFORE reg: NCC={ncc(fa, ma):.4f}  MSE={mse(fa, ma):.6f}")
print(f"warped AFTER  reg: NCC={ncc(fa, wa):.4f}  MSE={mse(fa, wa):.6f}")
print(f"array shape (LR, AP, SI) = {fa.shape}   # axis 2 = head-foot")


def best_slices(mask):
    """Index of the most-informative slice along each axis (max foreground)."""
    out = []
    for ax in range(3):
        other = tuple(i for i in range(3) if i != ax)
        out.append(int(np.argmax(mask.sum(axis=other))))
    return out


slices = best_slices(fa > (fa.mean() + 0.5 * fa.std()))

# (title, slice axis, slice idx, vertical-label, horizontal-label)
planes = [
    ("axial (perp. to SI / head-foot)", 2, slices[2], "AP", "L↔R"),
    ("coronal (perp. to AP)", 1, slices[1], "SI", "L↔R"),
    ("sagittal (perp. to LR)", 0, slices[0], "SI", "A↔P"),
]

fig, axs = plt.subplots(3, 4, figsize=(16, 13))
for row, (name, axis, sidx, vy, hx) in enumerate(planes):
    f = np.take(fa, sidx, axis=axis)
    m = np.take(ma, sidx, axis=axis)
    w = np.take(wa, sidx, axis=axis)
    overlay = np.stack([f, w, np.zeros_like(f)], axis=-1)  # R=fixed, G=warped -> yellow=aligned
    for col, (im, title, cmap) in enumerate([
        (f, f"fixed [{name}]", "gray"),
        (m, f"moving BEFORE [{name}]", "gray"),
        (w, f"warped AFTER [{name}]", "gray"),
        (overlay, "overlay fixed(R)+warped(G)", None),
    ]):
        ax = axs[row, col]
        ax.imshow(im, cmap=cmap, aspect="equal") if cmap else ax.imshow(im, aspect="equal")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(hx, fontsize=8)
        ax.set_ylabel(vy, fontsize=8)
        ax.tick_params(labelsize=7)

plt.suptitle(
    "ConvexAdam validation  |  de-skewed to anatomical axes (z/SI = head-foot)\n"
    f"recover known 45deg rotation + 20mm shift   "
    f"BEFORE NCC={ncc(fa, ma):.3f} / MSE={mse(fa, ma):.4f}   "
    f"AFTER NCC={ncc(fa, wa):.3f} / MSE={mse(fa, wa):.4f}",
    fontsize=12,
)
plt.tight_layout(rect=(0, 0, 1, 0.95))
out_png = FIG / "validate_known_transform.png"
plt.savefig(out_png, dpi=110, bbox_inches="tight")
print(f"saved: {out_png}")
