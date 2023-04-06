import logging
import numpy as np
from IsoNet.preprocessing.cubes import create_cube_seeds,crop_cubes
from IsoNet.util.rotations import rotation_list
from IsoNet.preprocessing.simulate import apply_wedge
from IsoNet.preprocessing.simulate import apply_wedge_dcube
import mrcfile
import scipy
from multiprocessing import Pool
from functools import partial
from IsoNet.util.utils import mkfolder
import skimage
from IsoNet.preprocessing.img_processing import normalize
import os

import sys

def crop_to_size(array, crop_size, cube_size):
        start = crop_size//2 - cube_size//2
        end = crop_size//2 + cube_size//2
        return array[start:end,start:end,start:end]
def fsc_filter(map,fsc3d,limit_r):
    crop_size = map.shape[0]
    
    mw = fsc3d#skimage.transform.resize(fsc3d, [crop_size,crop_size,crop_size], order = 0)

    half_size = crop_size // 2 
    if limit_r <1 and limit_r>0:
        for i in range(crop_size):
            for j in range(crop_size):
                for k in range(crop_size):
                    r = ((i-half_size)**2 + (j-half_size)**2 + (k-half_size)**2)**0.5
                    if r > limit_r * half_size:
                        mw[i,j,k] = 1

    #mw[mw>0.5] =1
    #mw[mw<=0.5] =0
    #with mrcfile.new("fouriermask_full.mrc",overwrite=True) as mrc:
    #    mrc.set_data(mw)
    mw = np.fft.fftshift(mw)
    f_data = np.fft.fftn(map)
    #f_data[mw<0.5]=0
    f_data = f_data*mw
    #with mrcfile.new("fourier_filtered_full1.mrc",overwrite=True) as mrc:
    #    mrc.set_data(np.abs(np.fft.fftshift(f_data)).astype(np.float32))
    inv = np.fft.ifftn(f_data)
    outData = np.real(inv).astype(np.float32)
    #with mrcfile.new("fourier_filtered_full2.mrc",overwrite=True) as mrc:
    #    mrc.set_data(np.real(inv).astype(np.float32))


    return outData



def rescale_fsc(fsc3d, limit_r, crop_size, weighting = False):
    fsc3d = skimage.transform.resize(fsc3d, [crop_size,crop_size,crop_size], order = 0)
    fsc3d[fsc3d<0] = 0

    half_size = crop_size // 2 
    for i in range(crop_size):
        for j in range(crop_size):
            for k in range(crop_size):
                r = ((i-half_size)**2 + (j-half_size)**2 + (k-half_size)**2)**0.5
                if r > limit_r * half_size:
                    fsc3d[i,j,k] = 1
                #if r < 0.7* limit_r * half_size:
                #    fsc3d[i,j,k] = 1
                #d = r-(limit_r*half_size) #30
                #fsc3d[i,j,k]=max(1/(1+np.exp(-d)), fsc3d[i,j,k])
    fsc3d[half_size,half_size,half_size]=1
    #fsc3d[fsc3d>0.143] =1
    
    #fsc3d[fsc3d<=0.143] =0
    with mrcfile.new("fouriermask.mrc",overwrite=True) as mrc:
        mrc.set_data(fsc3d)
    return fsc3d

def get_cubes(mw3d, data_dir, crop_size, cube_size, noise_scale,noise_mean, prefix, inp):
    '''
    current iteration mrc(in the 'results') + infomation from orignal subtomo
    normalized predicted + normalized orig -> normalize
    rotate by rotation_list and feed to get_cubes_one
    '''

    #mw = "fouriermask.mrc"
    mrc, start = inp
    root_name = mrc.split('/')[-1].split('.')[0]


    #if iter_count > 1:
    #    current_mrc = '{}/{}_iter{:0>2d}.mrc'.format(result_dir,root_name,iter_count-1)
    #    with mrcfile.open(current_mrc) as mrcData:
    #        ow_data = mrcData.data.astype(np.float32)
    #    orig_data = ow_data#apply_wedge(ow_data, ld1=0, ld2=1, mw3d=mw3d) + iw_data
    #else:
    if True:
        with mrcfile.open(mrc) as mrcData:
            iw_data = mrcData.data.astype(np.float32)
        orig_data = iw_data

    #current_mask = "{}/mask{}.mrc".format(output_dir,root_name.split("subvolume")[1])
    #with mrcfile.open(current_mask) as mrcData:
    #    mask_data = mrcData.data.astype(np.float32)
    #print(current_mask)

    rotated_data = np.zeros((len(rotation_list), *orig_data.shape))
    #rotated_mask = np.zeros((len(rotation_list), *orig_data.shape))

    old_rotation = False # should be false
    if old_rotation:
        for i,r in enumerate(rotation_list):
            data = np.rot90(orig_data, k=r[0][1], axes=r[0][0])
            data = np.rot90(data, k=r[1][1], axes=r[1][0])
            rotated_data[i] = data
    else:
        from scipy.ndimage import affine_transform
        from scipy.stats import special_ortho_group 
        for i in range(len(rotation_list)):
            rot = special_ortho_group.rvs(3)
            center = (np.array(orig_data.shape) -1 )/2.
            offset = center-np.dot(rot,center)
            rotated_data[i] = affine_transform(orig_data,rot,offset=offset,mode = 'nearest')
            #rotated_mask[i] = affine_transform(mask_data,rot,offset=offset,mode = 'nearest')
    #print(np.std(rotated_data))
    
    datax = apply_wedge_dcube(rotated_data, mw3d=mw3d)
    #rotated_mask = apply_wedge_dcube(rotated_mask, mw3d=mw3d)
    #print(np.std(datax))    

    noise_a = np.random.normal(size = rotated_data.shape)
    #noise_a = apply_wedge_dcube(noise_a, mw3d=mw3d, ld1 = 0, ld2 = 1)

    for i in range(len(rotation_list)): 
        #data_mask = crop_to_size(rotated_mask[i], crop_size, cube_size)

        data_X = crop_to_size(datax[i], crop_size, cube_size)
        data_Y = crop_to_size(rotated_data[i], crop_size, cube_size)
        noise = crop_to_size(noise_a[i], crop_size, cube_size)

        #data_X = data_X*data_mask + data_Y*(1-data_mask) + noise* noise_scale
        data_X = data_X + noise* noise_scale
        #print(np.average(data_Y[data_mask<0.5]))
        #data_Y = data_Y*data_mask + noise_mean * (1-data_mask)# + data_Y*(1-data_mask)# + 0.5*data_Y*(1-data_mask)
        with mrcfile.new('{}/train_x/x{}_{}.mrc'.format(data_dir,prefix, start), overwrite=True) as output_mrc:
            output_mrc.set_data(data_X.astype(np.float32))
        with mrcfile.new('{}/train_y/y{}_{}.mrc'.format(data_dir,prefix, start), overwrite=True) as output_mrc:
            output_mrc.set_data(data_Y.astype(np.float32))
        start += 1



def get_cubes_list(mw3d, mrc_list, data_dir,output_dir, crop_size, cube_size, noise_scale = 1, noise_mean = 0, prefix=''):
    '''
    generate new training dataset:
    map function 'get_cubes' to mrc_list from subtomo_dir
    seperate 10% generated cubes into test set.
    '''

    dirs_tomake = ['train_x','train_y', 'test_x', 'test_y']
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for d in dirs_tomake:
        folder = '{}/{}'.format(data_dir, d)
        if not os.path.exists(folder):
            os.makedirs(folder)


    inp=[]
    for i,mrc in enumerate(mrc_list):
        inp.append((mrc, i*len(rotation_list)))    
    # inp: list 0f (mrc_dir, index * rotation times)

    preprocessing_ncpus = 16
    if preprocessing_ncpus > 1:
        func = partial(get_cubes, mw3d, data_dir, crop_size, cube_size,noise_scale, noise_mean, prefix)
        with Pool(preprocessing_ncpus) as p:
            p.map(func,inp)
    else:
        for i in inp:
            logging.info("{}".format(i))
            get_cubes(i, data_dir, crop_size, cube_size)

def split_train_test(data_dir,batch_size=8):
    all_path_x = os.listdir(data_dir+'/train_x')
    num_test = int(len(all_path_x) * 0.1) 
    num_test = num_test - num_test%batch_size + batch_size
    #all_path_y = ['y'+prefix+'_'+i.split('_')[1] for i in all_path_x ]
    all_path_y = ['y'+i[1:] for i in all_path_x ]
    #print(all_path_y)
    ind = np.random.permutation(len(all_path_x))[0:num_test]
    for i in ind:
        os.rename('{}/train_x/{}'.format(data_dir, all_path_x[i]), '{}/test_x/{}'.format(data_dir, all_path_x[i]) )
        os.rename('{}/train_y/{}'.format(data_dir, all_path_y[i]), '{}/test_y/{}'.format(data_dir, all_path_y[i]) )

def extract_subvolume(current_map, n_subvolume, crop_size, mask, output_dir, prefix=''):
    #extract subvolume
    #print(len(mask))
    seeds=create_cube_seeds(current_map, n_subvolume, crop_size, mask)
    subtomos=crop_cubes(current_map,seeds,crop_size)
    #submasks=crop_cubes(mask,seeds, crop_size)
    mrc_list = []
    for j,s in enumerate(subtomos):
        im_name = '{}/subvolume{}_{:0>6d}.mrc'.format(output_dir, prefix, j)
        with mrcfile.new(im_name, overwrite=True) as output_mrc:
            output_mrc.set_data(s.astype(np.float32))
        #msk_name = '{}/mask{}_{:0>6d}.mrc'.format(output_dir,prefix, j)
        #with mrcfile.new(msk_name, overwrite=True) as outputa_mrc:
        #    output_mrc.set_data(submasks[j].astype(np.float32))
        mrc_list.append(im_name)
    return mrc_list

def map_refine(halfmap, mask, fsc3d, voxel_size, limit_res, output_dir = "results", output_base="half1", n_subvolume = 50, cube_size = 64, crop_size = 96, weighting=False):
    log_level = "info"
    if log_level == "debug":
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])
    else:
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
        datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
        #logging.basicConfig(format='%(asctime)s.%(msecs)03d, %(levelname)-8s %(message)s',
        #datefmt="%Y-%m-%d,%H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])

    # Fixed parameters
    full_size = halfmap.shape[0]
    num_iterations = 5
    data_dir = output_dir+"/data"
    mkfolder(data_dir)
    fsc3d = fsc3d.copy()
    fsc3d = skimage.transform.resize(fsc3d, [full_size,full_size,full_size], order = 0)

    fsc3d[fsc3d<0] = 0
    fsc3d[fsc3d<0.143] = 0
    fsc3d[fsc3d>=0.143] = 1
    limit_r = (2*voxel_size)/limit_res
 
    if weighting:
        logging.info("Apply sqrt(2*FSC/(1+FSC)) for 3DFSC")
        fsc3d = np.sqrt((2*fsc3d)/(1+fsc3d))
    else:
        logging.info("No sqrt(2*FSC/(1+FSC)) for 3DFSC")


    fsc3d_full = fsc3d.copy()
    fsc3d = rescale_fsc(fsc3d, limit_r, crop_size, weighting)

    halfmap = normalize(halfmap)#, pmin=0, pmax=100)
    halfmap_origional = halfmap.copy()

    limit_r = (2*voxel_size)/limit_res
    halfmap = fsc_filter(halfmap, fsc3d_full, limit_r)
    fsc3d = rescale_fsc(fsc3d, limit_r, crop_size, weighting)

    from IsoNet.models.network import Net
    network = Net()

    current_map = halfmap
    noise_scale=np.std(halfmap[mask<0.1])*3
    noise_mean=np.average(halfmap[mask<0.1])

    #main iterations
    for iter_count in range(1,num_iterations+1):
        
        previous_filename = "{}/corrected_{}_iter{}.mrc".format(output_dir, output_base, iter_count-1)

        if iter_count > 1:
            with mrcfile.open(previous_filename, 'r') as mrc:
                current_map = mrc.data
        else:
            with mrcfile.new(previous_filename, overwrite=True) as mrc:
                mrc.set_data(current_map)

        current_map = normalize(current_map)
        mrc_list = extract_subvolume(current_map, n_subvolume, crop_size, mask, output_dir)


        logging.info("Start Iteration{}!".format(iter_count))

        logging.info("Start preparing subvolumes!")
        get_cubes_list(fsc3d, mrc_list, data_dir, output_dir, crop_size, cube_size, noise_scale = noise_scale, noise_mean = noise_mean, prefix='')
        split_train_test(data_dir,batch_size=8)
        logging.info("Done preparing subvolumes!")

        ### start training and save model and json ###
        logging.info("Start training!")

        metrics = network.train(data_dir,gpuID=0, batch_size=8, epochs = 5, steps_per_epoch = 200, acc_grad = False) #train based on init model and save new one as model_iter{num_iter}.h5
        logging.info("Start predicting!")
        #network.predict(mrc_list, result_dir, iter_count+1, mw3d=fsc3d)
        #logging.info("Done predicting subvolumes!")

        #logging.info("Done training!")
        #current_filename_n = "{}/corrected_norm_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
        current_filename = "{}/corrected_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
        current_filename1 = "{}/corrected1_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
        current_filename2 = "{}/corrected2_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
        outData = network.predict_map(halfmap, fsc3d_full, fsc3d, output_file=current_filename, voxel_size = voxel_size )

                #outData = normalize(outData,percentile=args.normalize_percentile)
        #with mrcfile.new(current_filename, overwrite=True) as output_mrc:
        #    output_mrc.set_data(outData.astype(np.float32))
        #    output_mrc.voxel_size = voxel_size
        # with mrcfile.new(current_filename1, overwrite=True) as output_mrc:
        #     output_mrc.set_data((outData+halfmap).astype(np.float32))
        #     output_mrc.voxel_size = voxel_size
        # with mrcfile.new(current_filename2, overwrite=True) as output_mrc:
        #     output_mrc.set_data((outData+halfmap_origional).astype(np.float32))
        #     output_mrc.voxel_size = voxel_size

        logging.info('Done predicting')
        #network.predict_map(normalize(halfmap), fsc3d, output_file=current_filename_n, voxel_size = voxel_size )



def map_refine_multi(halfmap, mask, fsc3d, voxel_size, limit_res, output_dir = "results", output_base="half1", n_subvolume = 50, cube_size = 64, crop_size = 96, weighting=False):
    log_level = "info"
    if log_level == "debug":
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])
    else:
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
        datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
        #logging.basicConfig(format='%(asctime)s.%(msecs)03d, %(levelname)-8s %(message)s',
        #datefmt="%Y-%m-%d,%H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])

    n_sub_list = []
    count = 0
    for i,m in enumerate(mask):
        n = np.count_nonzero(mask[i]>0.5)
        count += n
        n_sub_list.append(n)
    logging.info(n_sub_list)
    for i in range(len(n_sub_list)):
        n_sub_list[i] = int(n_sub_list[i]/count * n_subvolume) + 1
    logging.info(n_sub_list)

    # Fixed parameters
    full_size = halfmap[0].shape[0]
    num_iterations = 5
    data_dir = output_dir+"/data"
    mkfolder(data_dir)
    fsc3d = fsc3d.copy()
    fsc3d = skimage.transform.resize(fsc3d, [full_size,full_size,full_size], order = 0)

    fsc3d[fsc3d<0] = 0
    fsc3d[fsc3d<0.143] = 0
    fsc3d[fsc3d>=0.143] = 1
    limit_r = (2*voxel_size)/limit_res
 
    if weighting:
        logging.info("Apply sqrt(2*FSC/(1+FSC)) for 3DFSC")
        fsc3d = np.sqrt((2*fsc3d)/(1+fsc3d))
    else:
        logging.info("No sqrt(2*FSC/(1+FSC)) for 3DFSC")


    fsc3d_full = fsc3d.copy()
    fsc3d = rescale_fsc(fsc3d, limit_r, crop_size, weighting)

    #with Pool(len(halfmap)) as p:
    #    halfmap = p.map(f, halfmap)
    for i,h in enumerate(halfmap):
        halfmap[i] = normalize(h)#, pmin=0, pmax=100)
        halfmap[i] = fsc_filter(halfmap[i], fsc3d_full, limit_r)    

    from IsoNet.models.network import Net
    network = Net()

    #main iterations
    for iter_count in range(1,num_iterations+1):
        mkfolder(data_dir)
        for i,h in enumerate(halfmap):
            previous_filename = "{}/corrected{}_{}_iter{}.mrc".format(output_dir, i, output_base, iter_count-1)
            print(previous_filename)

            if iter_count > 1:
                with mrcfile.open(previous_filename, 'r') as mrc:
                    current_map = mrc.data
                current_map = normalize(current_map)
            else:
                with mrcfile.new(previous_filename, overwrite=True) as mrc:
                    mrc.set_data(halfmap[i])
                current_map = normalize(halfmap[i])
            
            mrc_list = extract_subvolume(current_map, n_sub_list[i], crop_size, mask[i], output_dir)#,prefix=str(i))
            #print(mrc_list)
            noise_scale=np.std(halfmap[i][mask[i]<0.1])*0
            noise_mean=np.average(halfmap[i][mask[i]<0.1])

            logging.info("Start Iteration{}!".format(iter_count))

            logging.info("Start preparing subvolumes!")
            get_cubes_list(fsc3d, mrc_list, data_dir, output_dir, crop_size, cube_size, noise_scale = noise_scale, noise_mean = noise_mean)#,prefix=str(i))
            logging.info("Done preparing subvolumes!")
        split_train_test(data_dir,batch_size=8)

        ### start training and save model and json ###
        logging.info("Start training!")

        metrics = network.train(data_dir,gpuID=0, batch_size=8, epochs = 5, steps_per_epoch = 200, acc_grad = False) #train based on init model and save new one as model_iter{num_iter}.h5
        logging.info("Start predicting!")
        #network.predict(mrc_list, result_dir, iter_count+1, mw3d=fsc3d)
        #logging.info("Done predicting subvolumes!")

        #logging.info("Done training!")
        #current_filename_n = "{}/corrected_norm_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
        #current_filename = "{}/corrected_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
        #current_filename1 = "{}/corrected1_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
        #current_filename2 = "{}/corrected2_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
        #outData = network.predict_map(halfmap, fsc3d_full, fsc3d, output_file=current_filename, voxel_size = voxel_size )
        for i,h in enumerate(halfmap):
            current_filename = "{}/corrected{}_{}_iter{}.mrc".format(output_dir,i, output_base, iter_count) 
            current_filename1 = "{}/corrected1_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
            current_filename2 = "{}/corrected2_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
            network.predict_map(halfmap[i], fsc3d_full, fsc3d, output_file=current_filename, voxel_size = voxel_size )
                #outData = normalize(outData,percentile=args.normalize_percentile)
        #with mrcfile.new(current_filename, overwrite=True) as output_mrc:
        #    output_mrc.set_data(outData.astype(np.float32))
        #    output_mrc.voxel_size = voxel_size
        # with mrcfile.new(current_filename1, overwrite=True) as output_mrc:
        #     output_mrc.set_data((outData+halfmap).astype(np.float32))
        #     output_mrc.voxel_size = voxel_size
        # with mrcfile.new(current_filename2, overwrite=True) as output_mrc:
        #     output_mrc.set_data((outData+halfmap_origional).astype(np.float32))
        #     output_mrc.voxel_size = voxel_size

        logging.info('Done predicting')
        #network.predict_map(normalize(halfmap), fsc3d, output_file=current_filename_n, voxel_size = voxel_size )
    
