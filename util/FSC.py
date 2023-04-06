#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import mrcfile
from numpy.fft import fftshift, fftn
#from scipy.ndimage import zoom
from skimage.transform import resize
from scipy.spatial.distance import pdist,squareform

def get_FSC_map(half_maps):
    f1 = fftshift(fftn(halfmaps[0]))
    f2 = fftshift(fftn(halfmaps[1]))
    ret = np.real(np.multiply(f1,np.conj(f2)))
    n1 = np.real(np.multiply(f1,np.conj(f1)))
    n2 = np.real(np.multiply(f2,np.conj(f2)))
    FSC_map = ret/np.sqrt(n1*n2)
    return FSC_map

def rotational_average(input_map):
    nz,ny,nx = input_map.shape
    r = np.arange(nz)-nz//2

    [Z,Y,X] = np.meshgrid(r,r,r)
    index = np.round(np.sqrt(Z**2+Y**2+X**2))

    FSC_curve = np.zeros(nz//2)
    for i in range(nz//2):
        FSC_curve[i] = np.average(input_map[index==i])
    return FSC_curve
    
def ThreeD_FSC(FSC_map, angle=20, resize_to=96):
    nz_origional,ny,nx = FSC_map.shape

    FSC_map = resize(FSC_map,[resize_to, resize_to, resize_to])

    threshold = np.cos(np.deg2rad(angle))
    nz = resize_to
    r = np.arange(nz)-nz//2
    [Z,Y,X] = np.meshgrid(r,r,r)
    index = np.round(np.sqrt(Z**2+Y**2+X**2))

    out = np.zeros_like(FSC_map)
    out[nz//2,nz//2,nz//2] = 1
    
    for i in range(1,nz//2):
        pixels_T = np.where(index==i)
        pixels = np.transpose(pixels_T)
        dist_mat = squareform(pdist(pixels-nz//2, 'cosine'))
        FSC_values = FSC_map[pixels_T]
        f = lambda i : np.average(FSC_values[dist_mat[i]<1-threshold])
        out_list = list(map(f, np.arange(len(pixels))))
        out[pixels_T] = out_list
    out = resize(out,[nz_origional, nz_origional, nz_origional])
    return out    

if __name__ == '__main__':
        
    fNHalfMap1='emd_8731_half_map_1.mrc'
    fNHalfMap2='emd_8731_half_map_2.mrc'

    halfmaps = []
    with mrcfile.open(fNHalfMap1,'r') as mrc:
        halfmaps.append(mrc.data)
    with mrcfile.open(fNHalfMap2,'r') as mrc:
        halfmaps.append(mrc.data)

    FSC_map = get_FSC_map(halfmaps)
    out = ThreeD_FSC(FSC_map)
    print('here')
    with mrcfile.new('FSC.mrc',overwrite=True) as mrc:
        mrc.set_data(FSC_map.astype(np.float32))   
    with mrcfile.new('3DFSC.mrc',overwrite=True) as mrc:
        mrc.set_data(out.astype(np.float32))   

