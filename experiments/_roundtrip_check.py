"""Sanity check: nib_to_sitk / sitk_to_nib round-trip + agreement with sitk.ReadImage."""
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imaging_utils import nib_to_sitk, sitk_to_nib

p = Path(__file__).resolve().parent.parent / "chaos_data" / "train" / "1" / "T2SPIR.nii.gz"
nib_img = nib.load(str(p))
arr, aff = np.asarray(nib_img.dataobj, np.float32), nib_img.affine
ref = sitk.ReadImage(str(p))
mine = nib_to_sitk(arr, aff)

print("nibabel orient:", nib.aff2axcodes(aff), " shape:", arr.shape)
print("spacing  ref=", tuple(round(s, 4) for s in ref.GetSpacing()))
print("spacing mine=", tuple(round(s, 4) for s in mine.GetSpacing()))
print("origin   ref=", tuple(round(o, 3) for o in ref.GetOrigin()))
print("origin  mine=", tuple(round(o, 3) for o in mine.GetOrigin()))
print("direction ref=", tuple(round(d, 4) for d in ref.GetDirection()))
print("direction mine=", tuple(round(d, 4) for d in mine.GetDirection()))

arr2, aff2 = sitk_to_nib(mine)
print("arr round-trip equal:", np.allclose(arr, arr2, atol=1e-4),
      "maxdiff=", float(np.abs(arr - arr2).max()))
print("aff round-trip equal:", np.allclose(aff, aff2, atol=1e-3))
