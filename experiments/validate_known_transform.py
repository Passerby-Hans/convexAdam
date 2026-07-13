"""
Validate ConvexAdam registration on the bundled anisotropic test.

The upstream test ``test_convex_adam_rotated_and_shifted_anisotropic`` takes the
t2w volume, applies a KNOWN transform (45 deg rotation + 20 mm translation) to
create the moving image, then registers moving -> fixed. We compare alignment
BEFORE vs AFTER registration against the fixed (ground-truth pose) image.

Why a known transform (not frame-drop): registration aligns two misaligned
images, so its correct ground-truth check is "did we recover the known
displacement?". Frame-drop (undersampling) validates super-resolution, not
registration -- that experiment belongs to ECLARE.

Reads from ConvexAdam/tests/output/10000/ (produced by running the test).
Writes figures to ConvexAdam/experiments/figures/.

Run (from WSL, convexadam venv, cwd = ConvexAdam root):
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


def arr(name):
    img = sitk.ReadImage(str(OUT / f"{SUBJ}_{name}.mha"))
    return sitk.GetArrayFromImage(img).astype(np.float32)


def norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def ncc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()) + 1e-12))


def mse(a, b):
    return float(((a - b) ** 2).mean())


fa = norm(arr("fixed_resampled"))
ma = norm(arr("moving_rotated_and_shifted_resampled"))
wa = norm(arr("moving_rotated_and_shifted_resampled_warped"))

print("=== alignment vs fixed (ground-truth pose) ===")
print(f"moving BEFORE reg: NCC={ncc(fa, ma):.4f}  MSE={mse(fa, ma):.6f}")
print(f"warped AFTER  reg: NCC={ncc(fa, wa):.4f}  MSE={mse(fa, wa):.6f}")
print(f"improvement: NCC +{ncc(fa, wa) - ncc(fa, ma):.4f}")

axes_names = ["axial (z)", "coronal (y)", "sagittal (x)"]
fig, axs = plt.subplots(3, 4, figsize=(16, 12))
for row, (name, axis) in enumerate(zip(axes_names, range(3))):
    sidx = fa.shape[axis] // 2
    f = np.take(fa, sidx, axis=axis)
    m = np.take(ma, sidx, axis=axis)
    w = np.take(wa, sidx, axis=axis)
    overlay = np.stack([f, w, np.zeros_like(f)], axis=-1)  # R=fixed G=warped -> yellow=aligned
    for col, (img, title, cmap) in enumerate([
        (f, f"fixed [{name}]", "gray"),
        (m, f"moving BEFORE [{name}]", "gray"),
        (w, f"warped AFTER [{name}]", "gray"),
        (overlay, "overlay fixed(R)+warped(G)", None),
    ]):
        ax = axs[row, col]
        ax.imshow(img, cmap=cmap) if cmap else ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

plt.suptitle(
    "ConvexAdam validation: recover known 45deg rotation + 20mm translation\n"
    f"BEFORE NCC={ncc(fa, ma):.3f} / MSE={mse(fa, ma):.4f}   "
    f"AFTER NCC={ncc(fa, wa):.3f} / MSE={mse(fa, wa):.4f}",
    fontsize=12,
)
plt.tight_layout(rect=(0, 0, 1, 0.96))
out_png = FIG / "validate_known_transform.png"
plt.savefig(out_png, dpi=110, bbox_inches="tight")
print(f"saved: {out_png}")
