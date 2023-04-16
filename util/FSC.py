#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import mrcfile
from numpy.fft import fftshift, fftn
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
from skimage.transform import resize

def get_FSC_map(halfmaps, mask):
    h1 = halfmaps[0] * mask
    h2 = halfmaps[1] * mask
    f1 = fftshift(fftn(h1))
    f2 = fftshift(fftn(h2))
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



def calculate_FSC(pixels_T, FSC_values, point_tree, r0):
    values = np.zeros(len(pixels_T[0]))
    for j, pixel in enumerate(zip(pixels_T[0], pixels_T[1], pixels_T[2])):
        values[j] = np.average(FSC_values[point_tree.query_ball_point(pixel, r0)])
    return values

def ThreeD_FSC(FSC_map, limit_r=None, angle=20, n_processes=16):
    nz, ny, nx = FSC_map.shape
    if limit_r is None:
        limit_r = nz//2
    
    #FSC_map = resize(FSC_map,[resize_to, resize_to, resize_to])

    threshold = 2 * np.sin(np.deg2rad(angle) / 2.0)

    r = np.arange(nz) - nz // 2
    [Z,Y,X] = np.meshgrid(r,r,r)
    index = np.round(np.sqrt(Z ** 2 + Y ** 2 + X ** 2))

    FSC_map = FSC_map.astype(np.float32)
    out = np.zeros_like(FSC_map)
    out[nz // 2, nz // 2, nz // 2] = 1

    with Pool(processes=n_processes) as pool:
        results = []
        for i in range(1, limit_r):
            pixels_T = np.where(index == i)
            pixels = np.transpose(pixels_T)
            point_tree = cKDTree(pixels)
            r0 = i * threshold
            FSC_values = FSC_map[pixels_T]

            results.append(pool.apply_async(calculate_FSC, (pixels_T, FSC_values, point_tree, r0)))

        for i, result in enumerate(results):
            pixels_T = np.where(index == i + 1)
            out[pixels_T] = result.get()

    #out = resize(out,[nz_origional, nz_origional, nz_origional])
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

