"""
Clean CHAOS MRI DICOM -> RAS NIfTI converter (written from scratch, not ported).

Reads the read-only CHAOS raw archive and writes axis-aligned RAS NIfTI volumes
to ConvexAdam/chaos_data/{split}/{case_id}/{series}.nii.gz.

Correctness notes (better than naive IPP[2] sorting):
  * slice normal from the cross product of the DICOM row/col direction cosines
    (ImageOrientationPatient);
  * slices sorted by projecting ImagePositionPatient onto that slice normal
    (robust for oblique acquisitions, not just axis-aligned);
  * through-plane direction & spacing taken from the *actual* IPP delta between
    consecutive sorted slices (matches real data ordering);
  * voxel->physical affine built from row_cos*row_spacing, col_cos*col_spacing,
    slice_dir*zs, origin = first-slice IPP;
  * output reoriented to canonical RAS via nib.as_closest_canonical, so axis 2
    is always Superior-Inferior (head-foot) regardless of acquisition.

CHAOS MR layout (per case):
  .../MR/<case>/T1DUAL/DICOM_anon/{InPhase,OutPhase}/*.dcm
  .../MR/<case>/T2SPIR/DICOM_anon/*.dcm

Run (WSL, convexadam venv, cwd = ConvexAdam root):
    python experiments/convert_chaos.py --split Train --cases 1 2
"""
import argparse
from pathlib import Path

import numpy as np
import pydicom
import nibabel as nib
from nibabel.orientations import (
    axcodes2ornt,
    ornt_transform,
    io_orientation,
    apply_orientation,
    inv_ornt_aff,
)

REPO = Path(__file__).resolve().parent.parent
RAW_ROOT = REPO.parent / "chaos-raw"          # read-only raw archive
OUT_ROOT = REPO / "chaos_data"                # converted volumes (gitignored)


def _scalar(ds, name, default=None):
    v = getattr(ds, name, None)
    return float(v) if v is not None else default


def load_series(dcm_dir: Path):
    """Read all DICOMs in a directory, return (volume[R,C,S], affine4x4)."""
    files = sorted(dcm_dir.glob("*.dcm"))
    if not files:
        return None
    datasets = [pydicom.dcmread(str(f), stop_before_pixels=False) for f in files]

    # direction cosines from first slice (constant across a series)
    iop = [float(x) for x in datasets[0].ImageOrientationPatient]
    row_cos = np.array(iop[:3])
    col_cos = np.array(iop[3:6])
    slice_normal = np.cross(row_cos, col_cos)
    slice_normal /= np.linalg.norm(slice_normal)

    # sort slices by position projected on the slice normal
    def pos(ds):
        ipp = [float(x) for x in ds.ImagePositionPatient]
        return float(np.dot(ipp, slice_normal))

    datasets.sort(key=pos)
    positions = [pos(ds) for ds in datasets]

    rs, cs = (float(x) for x in datasets[0].PixelSpacing)        # row, col spacing
    if len(datasets) >= 2:
        zs = abs(positions[1] - positions[0])
        slice_dir = np.array([float(x) for x in datasets[1].ImagePositionPatient]) - \
                    np.array([float(x) for x in datasets[0].ImagePositionPatient])
        slice_dir /= np.linalg.norm(slice_dir)
    else:
        zs = _scalar(datasets[0], "SliceThickness", rs)
        slice_dir = slice_normal

    # stack pixel data (apply rescale)
    slope = _scalar(datasets[0], "RescaleSlope", 1.0)
    intercept = _scalar(datasets[0], "RescaleIntercept", 0.0)
    slices = [ds.pixel_array.astype(np.float32) * slope + intercept for ds in datasets]
    volume = np.stack(slices, axis=-1)            # (R, C, S)

    first_ipp = np.array([float(x) for x in datasets[0].ImagePositionPatient])
    affine = np.eye(4)
    affine[:3, 0] = row_cos * rs
    affine[:3, 1] = col_cos * cs
    affine[:3, 2] = slice_dir * zs
    affine[:3, 3] = first_ipp

    return volume, affine


def to_ras(volume, affine):
    """Reorient (volume, affine) to canonical RAS via nibabel orientation transform."""
    ornt = io_orientation(affine)
    ras = axcodes2ornt("RAS")
    transform = ornt_transform(ornt, ras)
    vol_ras = apply_orientation(volume, transform)
    aff_ras = affine @ inv_ornt_aff(transform, volume.shape)
    return vol_ras.astype(np.float32), aff_ras


def process_case(case_id, split):
    base = RAW_ROOT / f"CHAOS_{split}_Sets" / f"{split}_Sets" / "MR" / str(case_id)
    if not base.exists():
        print(f"  SKIP case {case_id}: {base} not found")
        return False
    out_dir = OUT_ROOT / ("train" if split == "Train" else "test") / str(case_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = []
    t1 = base / "T1DUAL" / "DICOM_anon"
    if t1.exists():
        for ph in ("InPhase", "OutPhase"):
            d = t1 / ph
            if d.exists():
                plan.append((d, f"T1DUAL_{ph}"))
    t2 = base / "T2SPIR" / "DICOM_anon"
    if t2.exists() and list(t2.glob("*.dcm")):
        plan.append((t2, "T2SPIR"))

    for dcm_dir, name in plan:
        result = load_series(dcm_dir)
        if result is None:
            print(f"  SKIP {name}: no DICOM in {dcm_dir}")
            continue
        volume, affine = result
        vol_ras, aff_ras = to_ras(volume, affine)
        nii = nib.Nifti1Image(vol_ras, aff_ras)
        nib.save(nii, out_dir / f"{name}.nii.gz")
        ax = nib.aff2axcodes(aff_ras)
        print(f"  {name:18s} shape={vol_ras.shape} "
              f"zooms={tuple(round(z,3) for z in nii.header.get_zooms())} "
              f"orient={ax} -> {out_dir.name}/{name}.nii.gz")
    return True


def main():
    ap = argparse.ArgumentParser(description="Clean CHAOS DICOM -> RAS NIfTI")
    ap.add_argument("--split", default="Train", choices=["Train", "Test"])
    ap.add_argument("--cases", type=int, nargs="*", default=None,
                    help="case ids; default all in split")
    args = ap.parse_args()

    if args.cases is None:
        mr = RAW_ROOT / f"CHAOS_{args.split}_Sets" / f"{args.split}_Sets" / "MR"
        args.cases = sorted(int(p.name) for p in mr.iterdir() if p.is_dir() and p.name.isdigit())

    print(f"=== convert CHAOS {args.split}: cases {args.cases} -> {OUT_ROOT} ===")
    for c in args.cases:
        process_case(c, args.split)
    print("done.")


if __name__ == "__main__":
    main()
