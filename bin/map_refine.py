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
import sys

def crop_to_size(array, crop_size, cube_size):
        start = crop_size//2 - cube_size//2
        end = crop_size//2 + cube_size//2
        return array[start:end,start:end,start:end]

def rescale_fsc(fsc3d, limit_r, crop_size, weighting = False):
    import skimage
    fsc3d = skimage.transform.resize(fsc3d, [crop_size,crop_size,crop_size], order = 0)
    fsc3d[fsc3d<0] = 0
    if weighting:
        logging.info("Apply sqrt(2*FSC/(1+FSC)) for 3DFSC")
        fsc3d = np.sqrt((2*fsc3d)/(1+fsc3d))
    else:
        logging.info("No sqrt(2*FSC/(1+FSC)) for 3DFSC")
    half_size = crop_size // 2 
    for i in range(crop_size):
        for j in range(crop_size):
            for k in range(crop_size):
                r = ((i-half_size)**2 + (j-half_size)**2 + (k-half_size)**2)**0.5
                if r > limit_r * half_size:
                    fsc3d[i,j,k] = 1
                #d = r#30
                #fsc3d[i,j,k]=max(1/(1+np.exp(-d)), fsc3d[i,j,k])
    fsc3d[half_size,half_size,half_size]=1
    with mrcfile.new("fouriermask.mrc",overwrite=True) as mrc:
        mrc.set_data(fsc3d)
    return fsc3d

def get_cubes(mw3d, result_dir, data_dir, iter_count, crop_size, cube_size,  inp):
    '''
    current iteration mrc(in the 'results') + infomation from orignal subtomo
    normalized predicted + normalized orig -> normalize
    rotate by rotation_list and feed to get_cubes_one
    '''

    #mw = "fouriermask.mrc"
    mrc, start = inp
    root_name = mrc.split('/')[-1].split('.')[0]


    if iter_count > 1:
        current_mrc = '{}/{}_iter{:0>2d}.mrc'.format(result_dir,root_name,iter_count-1)
        with mrcfile.open(current_mrc) as mrcData:
            ow_data = mrcData.data.astype(np.float32)
        orig_data = ow_data#apply_wedge(ow_data, ld1=0, ld2=1, mw3d=mw3d) + iw_data
    else:
        with mrcfile.open(mrc) as mrcData:
            iw_data = mrcData.data.astype(np.float32)
        orig_data = iw_data

    current_mask = "{}/mask{}.mrc".format("subvolumes",root_name.split("subvolume")[1])
    with mrcfile.open(current_mask) as mrcData:
        mask_data = mrcData.data.astype(np.float32)
    #print(current_mask)

    rotated_data = np.zeros((len(rotation_list), *orig_data.shape))
    rotated_mask = np.zeros((len(rotation_list), *orig_data.shape))

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
            rotated_data[i] = affine_transform(orig_data,rot,offset=offset)
            rotated_mask[i] = affine_transform(mask_data,rot,offset=offset)
    
    datax = apply_wedge_dcube(rotated_data, mw3d=mw3d)

    for i in range(len(rotation_list)): 
        data_mask = crop_to_size(rotated_mask[i], crop_size, cube_size)

        data_X = crop_to_size(datax[i], crop_size, cube_size)
        data_Y = crop_to_size(rotated_data[i], crop_size, cube_size)
        data_X = data_X*data_mask + data_X*(1-data_mask)
        data_Y = data_Y*data_mask + data_Y*(1-data_mask)# + 0.5*data_Y*(1-data_mask)
        with mrcfile.new('{}/train_x/x_{}.mrc'.format(data_dir, start), overwrite=True) as output_mrc:
            output_mrc.set_data(data_X.astype(np.float32))
        with mrcfile.new('{}/train_y/y_{}.mrc'.format(data_dir, start), overwrite=True) as output_mrc:
            output_mrc.set_data(data_Y.astype(np.float32))
        start += 1



def get_cubes_list(mw3d, mrc_list, result_dir, data_dir, iter_count, crop_size, cube_size, batch_size=8):
    '''
    generate new training dataset:
    map function 'get_cubes' to mrc_list from subtomo_dir
    seperate 10% generated cubes into test set.
    '''
    import os
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
        func = partial(get_cubes, mw3d, result_dir, data_dir, iter_count, crop_size, cube_size)
        with Pool(preprocessing_ncpus) as p:
            p.map(func,inp)
    else:
        for i in inp:
            logging.info("{}".format(i))
            get_cubes(i, result_dir, data_dir, iter_count, crop_size, cube_size)

    all_path_x = os.listdir(data_dir+'/train_x')
    num_test = int(len(all_path_x) * 0.1) 
    num_test = num_test - num_test%batch_size + batch_size
    all_path_y = ['y_'+i.split('_')[1] for i in all_path_x ]
    ind = np.random.permutation(len(all_path_x))[0:num_test]
    for i in ind:
        os.rename('{}/train_x/{}'.format(data_dir, all_path_x[i]), '{}/test_x/{}'.format(data_dir, all_path_x[i]) )
        os.rename('{}/train_y/{}'.format(data_dir, all_path_y[i]), '{}/test_y/{}'.format(data_dir, all_path_y[i]) )

def map_refine(halfmap, mask, fsc3d, pixel_size, limit_res, output_file="half1.mrc", n_subvolume = 50, cube_size = 64, crop_size = 96, weighting=False):
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
    num_iterations = 5
    result_dir = "results"
    mkfolder(result_dir)
    subtomo_dir = "subvolumes"
    mkfolder(subtomo_dir)
    data_dir = result_dir+"/data"
    mkfolder(data_dir)

    # rescale fsc3d
    limit_r = (2*pixel_size)/limit_res
    fsc3d = rescale_fsc(fsc3d, limit_r, crop_size, weighting)

    #normalize
    from IsoNet.preprocessing.img_processing import normalize
    halfmap = normalize(halfmap)

    #extract subvolume
    seeds=create_cube_seeds(halfmap, n_subvolume, crop_size, mask)
    subtomos=crop_cubes(halfmap,seeds,crop_size)
    submasks=crop_cubes(mask,seeds, crop_size)
    mrc_list = []
    for j,s in enumerate(subtomos):
        im_name = '{}/subvolume_{:0>6d}.mrc'.format(subtomo_dir, j)
        with mrcfile.new(im_name, overwrite=True) as output_mrc:
            output_mrc.set_data(s.astype(np.float32))
        msk_name = '{}/mask_{:0>6d}.mrc'.format(subtomo_dir, j)
        with mrcfile.new(msk_name, overwrite=True) as output_mrc:
            output_mrc.set_data(submasks[j].astype(np.float32))
        mrc_list.append(im_name)
    from IsoNet.models.network import Net
    network = Net()

    #main iterations
    for iter_count in range(1,num_iterations+1):        
            logging.info("Start Iteration{}!".format(iter_count))

            logging.info("Start preparing subvolumes!")
            get_cubes_list(fsc3d, mrc_list, result_dir, data_dir, iter_count, crop_size, cube_size, batch_size=8)
            logging.info("Done preparing subvolumes!")

            ### start training and save model and json ###
            logging.info("Start training!")

            metrics = network.train(data_dir,gpuID=0, batch_size=8, epochs = 5, steps_per_epoch = 200, acc_grad = True) #train based on init model and save new one as model_iter{num_iter}.h5
            logging.info("Start predicting subvolumes!")
            network.predict(mrc_list, result_dir, iter_count+1, mw3d=fsc3d)
            logging.info("Done predicting subvolumes!")

            logging.info("Done training!")
    network.predict_map(halfmap, fsc3d, output_file=output_file)
    
    
    
