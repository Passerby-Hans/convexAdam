"""
Shared imaging utilities for CHAOS T1/T2 experiments.

Everything here works in **nibabel RAS** arrays (axis0=L-R, axis1=A-P, axis2=S-I)
so visualization is anatomically correct by construction. SimpleITK is used only
to feed ConvexAdam; nib_to_sitk / sitk_to_nib are exact inverses (verified by
round-trip) and handle the RAS<->LPS flip + (X,Y,Z)<->(Z,Y,X) reorder.

Written from scratch. No code ported from any other project.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, zoom

RAS_FLIP = np.diag([-1.0, -1.0, 1.0])  # LPS = FLIP @ RAS ; RAS = FLIP @ LPS


# ---------------------------------------------------------------------------
# nibabel RAS  <->  SimpleITK (LPS)  (exact inverses)
# ---------------------------------------------------------------------------

def nib_to_sitk(arr_ras: np.ndarray, affine_ras: np.ndarray) -> sitk.Image:
    """RAS numpy volume (X,Y,Z) -> SimpleITK image (correct LPS geometry)."""
    spacing = np.sqrt((affine_ras[:3, :3] ** 2).sum(axis=0))
    direction = (RAS_FLIP @ (affine_ras[:3, :3] / spacing)).flatten()
    origin = RAS_FLIP @ affine_ras[:3, 3]
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr_ras.transpose(2, 1, 0).astype(np.float32)))
    img.SetSpacing([float(s) for s in spacing])
    img.SetDirection([float(d) for d in direction])
    img.SetOrigin([float(o) for o in origin])
    return img


def sitk_to_nib(img: sitk.Image):
    """SimpleITK image -> (arr_ras (X,Y,Z) float32, affine_ras 4x4)."""
    arr = sitk.GetArrayFromImage(img).astype(np.float32).transpose(2, 1, 0)  # (X,Y,Z)
    spacing = np.array(img.GetSpacing(), float)
    d = np.array(img.GetDirection(), float).reshape(3, 3)
    origin_lps = np.array(img.GetOrigin(), float)
    affine = np.eye(4)
    affine[:3, :3] = RAS_FLIP @ d @ np.diag(spacing)
    affine[:3, 3] = RAS_FLIP @ origin_lps
    return arr, affine


# ---------------------------------------------------------------------------
# preprocessing (kept in nibabel RAS; affine updated correctly)
# ---------------------------------------------------------------------------

def spacing_of(affine):
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def resample_iso(arr, affine, target_spacing, order=1):
    """Resample to (near-)isotropic spacing (order=1 linear for images, 0 NN for
    labels). Returns (arr, affine) with affine columns rescaled."""
    cur = spacing_of(affine)
    factors = cur / np.array(target_spacing, float)
    out = zoom(arr.astype(np.float32), factors, order=order, mode="nearest")
    aff = affine.copy()
    aff[:3, :3] = affine[:3, :3] * (np.array(target_spacing, float) / cur)[None, :]
    return out, aff


def crop_or_pad(arr, target_shape, affine, pad_value=None):
    """Center crop or constant-pad to target_shape; shifts origin for crops.
    pad_value defaults to arr.min() (use 0 for labels)."""
    if pad_value is None:
        pad_value = float(arr.min())
    target = np.array(target_shape, int)
    out = arr
    aff = affine.copy()
    for ax in range(min(3, len(target))):
        d = out.shape[ax] - target[ax]
        if d > 0:
            start = d // 2
            out = np.take(out, range(start, start + target[ax]), axis=ax)
            if start > 0:
                aff[:3, 3] += aff[:3, ax] * start
        elif d < 0:
            before = (-d) // 2
            after = (-d) - before
            pad = [(0, 0)] * out.ndim
            pad[ax] = (before, after)
            out = np.pad(out, pad, mode="constant", constant_values=pad_value)
            aff[:3, 3] -= aff[:3, ax] * before
    return out, aff


# ---------------------------------------------------------------------------
# deformation: phi = affine + smooth elastic, BACKWARD warp (moving[p]=fixed[p+phi])
# ---------------------------------------------------------------------------

def _rand_rotation(device, max_deg=10.0):
    deg = (torch.rand(3, device=device) * 2.0 * max_deg - max_deg) * (np.pi / 180.0)
    cx, sx = torch.cos(deg[0]), torch.sin(deg[0])
    cy, sy = torch.cos(deg[1]), torch.sin(deg[1])
    cz, sz = torch.cos(deg[2]), torch.sin(deg[2])
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], device=device)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], device=device)
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], device=device)
    return Rz @ Ry @ Rx


def generate_phi(shape, sigma=8.0, magnitude=3.0, rot_deg=10.0, translation_vx=10.0, device="cpu"):
    """
    Displacement field phi (shape, 3) in voxel units = affine(rot<=rot_deg,
    scale 0.9-1.1, trans +-translation_vx) + Gaussian-smoothed elastic(sigma,
    magnitude).  Backward-warp convention: moving[p] = fixed[p + phi[p]].
    """
    X, Y, Z = shape
    dev = torch.device(device)
    g = torch.stack(torch.meshgrid(
        torch.arange(X, device=dev), torch.arange(Y, device=dev),
        torch.arange(Z, device=dev), indexing="ij"), dim=-1).float()
    center = torch.tensor([(X - 1) / 2, (Y - 1) / 2, (Z - 1) / 2], device=dev)
    R = _rand_rotation(dev, rot_deg)
    S = torch.diag(0.9 + torch.rand(3, device=dev) * 0.2)
    A = R @ S
    translation = torch.rand(3, device=dev) * (2.0 * translation_vx) - translation_vx
    coords_aff = center + ((g - center) @ A.T) + translation
    elastic = np.zeros((*shape, 3), np.float32)
    for c in range(3):
        n = np.random.randn(*shape).astype(np.float32)
        s = gaussian_filter(n, sigma=sigma, mode="reflect")
        m = np.abs(s).max()
        elastic[..., c] = s / m * magnitude if m > 1e-8 else s
    coords_def = coords_aff + torch.from_numpy(elastic).to(dev)
    return (coords_def - g).cpu().numpy().astype(np.float32)


def backward_warp(arr, phi):
    """moving[p] = fixed[p + phi[p]] (backward warp). arr,phi in (X,Y,Z)=(LR,AP,SI).

    Note: F.grid_sample's grid last axis is (x,y,z) = (W,H,D), i.e. REVERSED
    relative to the input's (D,H,W). We build the identity grid in (D,H,W) and
    reverse the last axis (+ phi's last axis) to match.
    """
    X, Y, Z = arr.shape  # LR, AP, SI counts
    img = torch.as_tensor(arr, dtype=torch.float32)[None, None]              # (1,1,D=X,H=Y,W=Z)
    base = torch.stack(torch.meshgrid(
        torch.arange(X), torch.arange(Y), torch.arange(Z), indexing="ij"),
        dim=-1)[None].float()                                               # (1,X,Y,Z,3) in (LR,AP,SI)
    phi_t = torch.as_tensor(phi, dtype=torch.float32)[None]                 # (1,X,Y,Z,3)
    sample = torch.flip(base, [-1]) + torch.flip(phi_t, [-1])               # -> (SI,AP,LR) order for grid_sample
    for i, n in enumerate([Z, Y, X]):                                       # sizes for (SI,AP,LR)
        sample[..., i] = 2.0 * sample[..., i] / max(n - 1, 1) - 1.0
    warped = F.grid_sample(img, sample, mode="bilinear", padding_mode="border", align_corners=True)
    return warped[0, 0].numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# metrics + anatomically-correct display
# ---------------------------------------------------------------------------

def ncc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()) + 1e-12))


def mse(a, b):
    return float(((a - b) ** 2).mean())


def ncc_roi(a, b, margin=8):
    """NCC over the interior (exclude a `margin`-voxel boundary band) so boundary
    sampling artefacts don't dominate."""
    s = (slice(margin, -margin),) * 3 if a.shape[0] > 2 * margin else (slice(None),) * 3
    return ncc(a[s], b[s])


def _slice_axes(axis):
    """Return (2D-array, vertical-label, horizontal-label) cut convention for a
    given RAS axis (0=LR,1=AP,2=SI). Display rule: imshow(slice.T, origin='lower')."""
    return {
        0: ("Sagittal", "I→S", "P→A"),   # slice along LR -> (AP, SI)
        1: ("Coronal",  "I→S", "L→R"),   # slice along AP -> (LR, SI)
        2: ("Axial",    "P→A", "L→R"),   # slice along SI -> (LR, AP)
    }[axis]


def save_comparison_png(fixed, moving, warped, out_png, title, slice_idx=None):
    """fixed/moving/warped are RAS arrays (X,Y,Z). 3 rows (axial/coronal/sagittal)
    x 4 cols (fixed | moving-before | warped-after | overlay R=fixed G=warped).
    Display: imshow(slice.T, origin='lower', aspect='equal')."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    f, m, w = norm(fixed), norm(moving), norm(warped)
    # pick informative slice per axis (most foreground in fixed)
    mask = f > (f.mean() + 0.5 * f.std())
    idx = [int(np.argmax(mask.sum(axis=tuple(k for k in range(3) if k != a)))) for a in range(3)]
    if slice_idx is not None:
        idx = [slice_idx] * 3

    fig, axs = plt.subplots(3, 4, figsize=(16, 13))
    for row, axis in enumerate([2, 1, 0]):  # axial, coronal, sagittal
        name, vy, hx = _slice_axes(axis)
        fs = np.take(f, idx[axis], axis=axis)
        ms = np.take(m, idx[axis], axis=axis)
        ws = np.take(w, idx[axis], axis=axis)
        ov = np.stack([fs, ws, np.zeros_like(fs)], axis=-1)
        for col, (im, ttl, cmap) in enumerate([
            (fs, f"fixed [{name}]", "gray"),
            (ms, f"moving BEFORE [{name}]", "gray"),
            (ws, f"warped AFTER [{name}]", "gray"),
            (ov, "overlay fixed(R)+warped(G)", None),
        ]):
            ax = axs[row, col]
            im2 = im.T if cmap else im.transpose(1, 0, 2)
            ax.imshow(im2, cmap=cmap, aspect="equal", origin="lower")
            ax.set_title(ttl, fontsize=9)
            ax.set_xlabel(hx, fontsize=8)
            ax.set_ylabel(vy, fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# label transfer: warp by ConvexAdam disp + per-organ Dice + label overlay
# ---------------------------------------------------------------------------

ORGANS = {63: "liver", 126: "spleen", 189: "R-kidney", 252: "L-kidney"}


def warp_ras_with_disp(arr_ras, disp_zyx, order=0):
    """Apply ConvexAdam backward displacement field (disp in (Z,Y,X,3)) to a RAS
    array (X,Y,Z). order=0 for discrete labels/FOV masks, 1 for images. Voxels
    sampled outside the source are 0 (so a ones-mask -> exact FOV coverage)."""
    from scipy.ndimage import map_coordinates
    arr_zyx = np.ascontiguousarray(arr_ras.transpose(2, 1, 0))
    d1, d2, d3, _ = disp_zyx.shape
    ident = np.meshgrid(np.arange(d1), np.arange(d2), np.arange(d3), indexing="ij")
    warped = map_coordinates(arr_zyx, disp_zyx.transpose(3, 0, 1, 2) + ident,
                             order=order, mode="constant", cval=0.0)
    return warped.transpose(2, 1, 0)


def dice_label(a, b, val, mask=None):
    """Dice for a single label value, optionally restricted to a boolean mask
    (e.g. the FOV-overlap region)."""
    if mask is not None:
        a = a[mask]
        b = b[mask]
    A = (a == val)
    B = (b == val)
    s = A.sum() + B.sum()
    return float(2.0 * (A & B).sum() / s) if s > 0 else float("nan")


def save_label_png(img, lab_true, lab_pred, overlap, out_png, dice_tbl, title,
                   labels, names=None):
    """Label-transfer figure (RAS arrays). axial slice = largest organ-bearing z.
    Panels: img + true labels | img + warped labels | agreement map.
    labels: list of label values; names: optional list of same length."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = _label_cmap(len(labels))
    names = names if names is not None else [str(v) for v in labels]

    z = int(np.argmax(((lab_true > 0) & overlap).sum(axis=(0, 1)))) \
        if ((lab_true > 0) & overlap).any() else img.shape[2] // 2

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    img_s = norm(img[:, :, z])
    true_s = _remap_labels(lab_true[:, :, z], labels)
    pred_s = _remap_labels(lab_pred[:, :, z], labels)

    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    for ax, ls, name in [(axs[0], true_s, "ground-truth label"),
                         (axs[1], pred_s, "warped label (after reg)")]:
        ax.imshow(img_s.T, cmap="gray", origin="lower", aspect="equal")
        ax.imshow(ls.T, cmap=cmap, vmin=0, vmax=len(labels), origin="lower",
                  aspect="equal", interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    agree = np.zeros(true_s.shape, int)
    agree[(true_s > 0) & (pred_s > 0) & (true_s == pred_s)] = 1
    agree[((true_s > 0) | (pred_s > 0)) & ~((true_s > 0) & (pred_s > 0) & (true_s == pred_s))] = 2
    axs[2].imshow(img_s.T, cmap="gray", origin="lower", aspect="equal")
    axs[2].imshow(agree.T, cmap=ListedColormap([(0, 0, 0, 0), (0.2, 0.9, 0.2, 0.6), (1, 0.2, 0.2, 0.6)]),
                  vmin=0, vmax=2, origin="lower", aspect="equal", interpolation="nearest")
    axs[2].set_title("green=agree  red=mismatch", fontsize=10)
    axs[2].axis("off")

    dice_str = "  ".join(f"{names[i]}={dice_tbl[v]:.2f}" for i, v in enumerate(labels))
    fig.suptitle(f"{title}\naxial z={z}  |  Dice (on FOV overlap): {dice_str}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _label_cmap(n):
    """Transparent background + n distinct colours (tab20)."""
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    base = plt.cm.tab20(np.linspace(0, 1, max(n, 1)))
    colors = [(0, 0, 0, 0)] + [(float(c[0]), float(c[1]), float(c[2]), 0.55) for c in base[:n]]
    return ListedColormap(colors)


def _remap_labels(a, labels):
    """Map each value in `labels` -> 1..N (0/background stays 0). Returns uint8."""
    out = np.zeros(a.shape, np.uint8)
    for i, lab in enumerate(labels):
        out[a == lab] = i + 1
    return out


def save_montage(img, lab, out_png, title, labels, cols=7):
    """Tile ALL axial (z) slices into a grid; each cell = image (gray) + labels
    (colour overlay). RAS arrays (X,Y,Z); z = SI = head-foot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    Z = img.shape[2]
    rows = (Z + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.1))
    axs = np.atleast_1d(axs).ravel()
    cmap = _label_cmap(len(labels))
    for z in range(Z):
        ax = axs[z]
        ax.imshow(norm(img[:, :, z]).T, cmap="gray", origin="lower", aspect="equal")
        ax.imshow(_remap_labels(lab[:, :, z], labels).T, cmap=cmap, vmin=0,
                  vmax=len(labels), origin="lower", aspect="equal", interpolation="nearest")
        ax.set_title(f"z={z}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for z in range(Z, len(axs)):
        axs[z].axis("off")
    fig.suptitle(f"{title}  ({Z} axial slices; {len(labels)} labels)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=90, bbox_inches="tight")
    plt.close(fig)


def save_warped_montage(t2_img, t2_lab, warped_img, warped_lab, overlap, out_png,
                        title, labels, pre_img=None, pre_lab=None, max_slices=36):
    """Per organ-bearing z-slice (subsampled), columns:
       [fixed img + true label] [+ pre-warp moving + label] [warped moving + label]
       [agreement map].
    pre_img/pre_lab (moving in fixed frame, BEFORE the disp warp) optional -> adds
    a "before registration" column. agreement: green=same label, red=one-sided."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = _label_cmap(len(labels))
    agree_cmap = ListedColormap([(0, 0, 0, 0), (0.2, 0.9, 0.2, 0.6), (1, 0.2, 0.2, 0.6)])
    vmax = len(labels)
    has_pre = pre_img is not None
    ncol = 4 if has_pre else 3

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    organz = [z for z in range(t2_img.shape[2])
              if ((t2_lab[:, :, z] > 0) & overlap[:, :, z]).any()]
    if not organz:
        organz = [t2_img.shape[2] // 2]
    step = max(1, (len(organz) + max_slices - 1) // max_slices)
    zs = organz[::step]

    rows = len(zs)
    fig, axs = plt.subplots(rows, ncol, figsize=(3.2 * ncol, rows * 2.05), squeeze=False)
    for r, z in enumerate(zs):
        ts = _remap_labels(t2_lab[:, :, z], labels)
        ps = _remap_labels(warped_lab[:, :, z], labels)
        col = 0
        # fixed
        ax = axs[r, col]; col += 1
        ax.imshow(norm(t2_img[:, :, z]).T, cmap="gray", origin="lower", aspect="equal")
        ax.imshow(ts.T, cmap=cmap, vmin=0, vmax=vmax, origin="lower", aspect="equal", interpolation="nearest")
        ax.set_title("fixed (target) + label", fontsize=8)
        # pre-warp moving (before registration)
        if has_pre:
            ax = axs[r, col]; col += 1
            qs = _remap_labels(pre_lab[:, :, z], labels)
            ax.imshow(norm(pre_img[:, :, z]).T, cmap="gray", origin="lower", aspect="equal")
            ax.imshow(qs.T, cmap=cmap, vmin=0, vmax=vmax, origin="lower", aspect="equal", interpolation="nearest")
            ax.set_title("moving BEFORE warp + label", fontsize=8)
        # warped moving (after registration)
        ax = axs[r, col]; col += 1
        ax.imshow(norm(warped_img[:, :, z]).T, cmap="gray", origin="lower", aspect="equal")
        ax.imshow(ps.T, cmap=cmap, vmin=0, vmax=vmax, origin="lower", aspect="equal", interpolation="nearest")
        ax.set_title("moving AFTER warp + label", fontsize=8)
        # agreement
        agree = np.zeros(ts.shape, int)
        agree[(ts > 0) & (ps > 0) & (ts == ps)] = 1
        agree[((ts > 0) | (ps > 0)) & ~((ts > 0) & (ps > 0) & (ts == ps))] = 2
        ax = axs[r, col]
        ax.imshow(norm(t2_img[:, :, z]).T, cmap="gray", origin="lower", aspect="equal")
        ax.imshow(agree.T, cmap=agree_cmap, vmin=0, vmax=2, origin="lower", aspect="equal", interpolation="nearest")
        ax.set_title(f"z={z}  green=agree red=mismatch", fontsize=8)
        for ax in axs[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=90, bbox_inches="tight")
    plt.close(fig)
