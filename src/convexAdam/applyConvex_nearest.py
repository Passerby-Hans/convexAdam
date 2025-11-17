import nibabel as nib
from scipy.ndimage.interpolation import map_coordinates
import numpy as np


import argparse



def main():

    parser = argparse.ArgumentParser()
    #inputdatagroup = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--input_field", dest="input_field", help="input convex displacement field (.nii.gz) full resolution", default=None, required=True)
    parser.add_argument("--input_moving", dest="input_moving",  help="input moving scan (.nii.gz)", default=None, required=True)
    parser.add_argument("--output_warped", dest="output_warped",  help="output waroed scan (.nii.gz)", default=None, required=True)


    options = parser.parse_args()
    d_options = vars(options)
    
    moving = nib.load(d_options['input_moving']).get_fdata().astype('float32')
    disp = nib.load(d_options['input_field']).get_fdata().astype('float32')
    H, W, D, _ = disp.shape
    identity = np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij')
    warped = map_coordinates(moving,disp.transpose(3,0,1,2)+identity,order=0)
    nii = nib.Nifti1Image(warped,None,header=nib.load(d_options['input_moving']).header)
    nib.save(nii,d_options['output_warped'])


if __name__ == '__main__':
    main()
