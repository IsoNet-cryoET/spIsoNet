#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import mrcfile
from numpy.fft import fftshift, fftn
from multiprocessing import Pool
import numpy as np
from scipy.spatial import cKDTree
from skimage.transform import resize
def recommended_resolution(fsc3d, voxel_size, threshold = 0.5):
    diameter = fsc3d.shape[0]
    center = diameter//2
    grid  = np.mgrid[0:diameter,0:diameter,0:diameter]
    r = ((grid[0]-center)**2 + (grid[1]-center)**2 + (grid[2]-center)**2)**0.5
    r = r.astype(np.int32).flatten()
    a = np.zeros(center, dtype = np.float32)
    df = fsc3d.flatten()
    for i in range(len(a)):        
        a[i] = np.average(df[r==i])
        if a[i] < threshold:
            return center/(i+1)*2 * voxel_size
    return 2 * voxel_size

def get_rayFSC(data, limit_r, n_sampling= 3, preserve_prec = 50):
    if preserve_prec < 5:
        return np.ones(data.shape, dtype =np.float32)
    
    nz = data.shape[0]

    rx = np.arange(nz//2+1)
    ry = np.arange(nz)-nz//2
    rz = np.arange(nz)-nz//2

    [Z,Y,X] = np.meshgrid(rz,ry,rx)

    rxy2 = X**2 + Y**2
    rxy = np.sqrt(X**2 + Y**2) + 1e-4
    rxyz = np.sqrt(rxy2 + Z**2) + 1e-4

    psi = np.arcsin(Y/rxy)
    tlt = np.arcsin(Z/rxyz)

    psi = (psi*n_sampling*2).astype(int)
    tlt = (tlt*n_sampling*2).astype(int)
    step = int(np.pi * n_sampling)

    import copy
    half_data = copy.deepcopy(data[:,:,nz//2-1:])
    for i in range(-step,step+1):
        print(f"{i}")
        for j in range(-step,step+1):
            arr_and = np.logical_and(psi==i,tlt==j)
            arr_and = np.logical_and(arr_and, rxyz <limit_r)
            if np.sum(arr_and) > 0:
                half_data[arr_and] = np.average(half_data[arr_and])

    useful_array = half_data[rxyz <limit_r]
    ma = np.percentile(useful_array,preserve_prec,keepdims=True)
    mi = np.percentile(useful_array,1,keepdims=True)
    useful_array = np.clip((useful_array - mi) / ( ma - mi + 1e-5 ), 0,1)
    half_data[rxyz <limit_r] = useful_array

    ray_FSC = np.zeros(data.shape)
    ray_FSC[:,:,nz//2-1:] = half_data
    ray_FSC[:,:,:nz//2-1] = np.flip(half_data[:,:,1:nz//2], axis = 2)    
    
    return ray_FSC

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
        values[j] = (FSC_values[point_tree.query_ball_point(pixel, r0)]).mean()
    return values

def ThreeD_FSC(FSC_map, limit_r=None, angle=20, n_processes=16):
    from tqdm import tqdm

    nz, ny, nx = FSC_map.shape
    if limit_r is None:
        limit_r = nz//2
    
    threshold = 2 * np.sin(np.deg2rad(angle) / 2.0)

    r = np.arange(nz) - nz // 2
    [Z,Y,X] = np.meshgrid(r,r,r)
    index = np.round(np.sqrt(Z ** 2 + Y ** 2 + X ** 2))

    FSC_map = FSC_map.astype(np.float32)
    out = np.zeros_like(FSC_map)
    out[nz // 2, nz // 2, nz // 2] = 1

    with Pool(processes=n_processes) as pool:
        results = []
        for i in tqdm(range(1, limit_r)):
            pixels_T = np.where(index == i)
            pixels = np.transpose(pixels_T)
            point_tree = cKDTree(pixels)
            r0 = i * threshold
            FSC_values = FSC_map[pixels_T]
            results.append(pool.apply_async(calculate_FSC, (pixels_T, FSC_values, point_tree, r0)))

        for i, result in enumerate(tqdm(results)):
            pixels_T = np.where(index == i + 1)
            out[pixels_T] = result.get()
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

