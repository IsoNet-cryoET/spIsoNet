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
from IsoNet.util.plot_metrics import plot_metrics
import shutil

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
        
def crop_to_size(array, crop_size, cube_size):
        start = crop_size//2 - cube_size//2
        end = crop_size//2 + cube_size//2
        return array[start:end,start:end,start:end]

def fsc_filter(map,fsc3d):
    mw = fsc3d
    mw = np.fft.fftshift(mw)
    f_data = np.fft.fftn(map)
    f_data = f_data*mw
    inv = np.fft.ifftn(f_data)
    outData = np.real(inv).astype(np.float32)
    return outData

def rescale_fsc(fsc3d, threshold, voxel_size, limit_res, crop_size):

    fsc3d = skimage.transform.resize(fsc3d, [crop_size,crop_size,crop_size])
    fsc3d[fsc3d<0] = 0

    half_size = crop_size // 2 
    limit_r = ((2*voxel_size)/limit_res) * half_size

    r = np.arange(crop_size)-crop_size//2
    [Z,Y,X] = np.meshgrid(r,r,r)
    index = np.round(np.sqrt(Z**2+Y**2+X**2))

    if threshold is not None:
        fsc3d[fsc3d<threshold] = 0
        fsc3d[fsc3d>threshold] = 1

    smooth_limit = True
    if smooth_limit:
        d =  index - limit_r
        #d[d<-10] = -10
        sigmoid = 1.0/(1.0+np.exp(-d))
        #sigmoid[sigmoid<1.0/(1.0+np.exp(-9))] = 0
        fsc3d = np.maximum(sigmoid.astype(np.float32), fsc3d)
    else:
        fsc3d[index > limit_r] = 1

    fsc3d[half_size,half_size,half_size] = 1
    return fsc3d


def get_cubes(mw3d, data_dir, crop_size, cube_size, noise_scale, inp):
    '''
    current iteration mrc(in the 'results') + infomation from orignal subtomo
    normalized predicted + normalized orig -> normalize
    rotate by rotation_list and feed to get_cubes_one
    '''
    mrc, start = inp

    if True:
        with mrcfile.open(mrc) as mrcData:
            iw_data = mrcData.data.astype(np.float32)
        orig_data = iw_data

    rotated_data = np.zeros((len(rotation_list), *orig_data.shape), dtype=np.float32)

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
    
    datax = apply_wedge_dcube(rotated_data, mw3d=mw3d)

    if noise_scale > 0:
        noise_a = np.random.normal(size = rotated_data.shape).astype(np.float32)
        noise_a = apply_wedge_dcube(noise_a, mw3d=mw3d, ld1 = 0, ld2 = 1)

    for i in range(len(rotation_list)): 
        data_X = crop_to_size(datax[i], crop_size, cube_size)
        data_Y = crop_to_size(rotated_data[i], crop_size, cube_size)
        
        
        # note change order here
        data_Y = data_Y# - data_X
        if noise_scale > 0:
            noise = crop_to_size(noise_a[i], crop_size, cube_size)
            data_X = data_X + noise / noise.std() * noise_scale


        with mrcfile.new('{}/train_x/x_{}.mrc'.format(data_dir, start), overwrite=True) as output_mrc:
            output_mrc.set_data(data_X.astype(np.float32))
        with mrcfile.new('{}/train_y/y_{}.mrc'.format(data_dir, start), overwrite=True) as output_mrc:
            output_mrc.set_data(data_Y.astype(np.float32))
        start += 1


def get_cubes_list(mw3d, mrc_list, data_dir,output_dir, crop_size, cube_size, noise_scale = 1):
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
    # inp: list of (mrc_dir, index * rotation times)

    preprocessing_ncpus = 12
    if preprocessing_ncpus > 1:
        func = partial(get_cubes, mw3d, data_dir, crop_size, cube_size, noise_scale)
        with Pool(preprocessing_ncpus) as p:
            p.map(func,inp)
    else:
        for item in inp:
            get_cubes(mw3d, data_dir, crop_size, cube_size, noise_scale, item)

def split_train_test(data_dir,batch_size=8):
    all_path_x = os.listdir(data_dir+'/train_x')
    num_test = int(len(all_path_x) * 0.1) 
    num_test = num_test - num_test%batch_size + batch_size
    #all_path_y = ['y_'+i.split('_')[1] for i in all_path_x ]
    all_path_y = ['y'+i[1:] for i in all_path_x ]
    ind = np.random.permutation(len(all_path_x))[0:num_test]
    for i in ind:
        os.replace('{}/train_x/{}'.format(data_dir, all_path_x[i]), '{}/test_x/{}'.format(data_dir, all_path_x[i]) )
        os.replace('{}/train_y/{}'.format(data_dir, all_path_y[i]), '{}/test_y/{}'.format(data_dir, all_path_y[i]) )

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



def map_refine(halfmap, mask, fsc3d, threshold, voxel_size, limit_res=None, 
               iterations = 10, epochs = 10, mixed_precision = False,
               output_dir = "results", output_base="half1", n_subvolume = 50, 
               cube_size = 64, crop_size = 96, predict_crop_size=128, noise_scale=None, batch_size = 8, acc_batches=2, gpuID="0", learning_rate= 4e-4):

    data_dir = output_dir+"/data"
    mkfolder(data_dir)

    fsc3d_cube = rescale_fsc(fsc3d, threshold, voxel_size, limit_res, crop_size)
    # with mrcfile.new('small_FSC.mrc', overwrite=True) as mrc:
    #     mrc.set_data(fsc3d_cube)
    fsc3d_full = rescale_fsc(fsc3d, threshold, voxel_size, limit_res, halfmap.shape[0])
    # with mrcfile.new('large_FSC.mrc', overwrite=True) as mrc:
    #     mrc.set_data(fsc3d_full)


    from IsoNet.models.network import Net
    #changed add_last fopr testing

    network = Net(filter_base = 64, add_last=True)

    # need to check normalize
    halfmap = normalize(halfmap,percentile=False)

    current_map = halfmap.copy()

    for iter_count in range(0,iterations+1):
        if iter_count == 0:
            current_filename = "{}/corrected_{}_iter{}.mrc".format(output_dir, output_base, iter_count)
            with mrcfile.new(current_filename, overwrite=True) as mrc:
                mrc.set_data(current_map)
                mrc.update_header_from_data()
                mrc._set_voxel_size(voxel_size,voxel_size,voxel_size)
            continue
        
        previous_filename = "{}/corrected_{}_iter{}.mrc".format(output_dir, output_base, iter_count-1)
        with mrcfile.open(previous_filename, 'r',permissive=True) as mrc:
            current_map = mrc.data
            current_map = normalize(current_map,percentile=False)
    
        mrc_list = extract_subvolume(current_map, n_subvolume, crop_size, mask, output_dir)

        logging.info("Start Iteration{}!".format(iter_count))

        logging.info("Start preparing subvolumes!")
        get_cubes_list(fsc3d_cube, mrc_list, data_dir, output_dir, crop_size, cube_size, noise_scale)
        split_train_test(data_dir,batch_size=batch_size)
        logging.info("Done preparing subvolumes!")

        logging.info("Start training!")
        #if iter_count > 1:
        #    network.load("{}/model_{}_iter{}.h5".format(output_dir, output_base, iter_count-1))
        network.train(data_dir, output_dir, batch_size=batch_size, epochs = epochs, steps_per_epoch = 1000, 
                                mixed_precision=mixed_precision, acc_batches=acc_batches, learning_rate = learning_rate) #train based on init model and save new one as model_iter{num_iter}.h5
        #network.save("{}/model_{}_iter{}.h5".format(output_dir, output_base, iter_count))
        plot_metrics(network.metrics, f"{output_dir}/loss_{output_base}.png")

        logging.info("Start predicting!")
        filtered_halfmap = fsc_filter(current_map, fsc3d_full)

        #with mrcfile.new(f"{output_dir}/filtered_halfmap{output_base}_iter{iter_count}.mrc", overwrite=True) as output_mrc:
        #    output_mrc.set_data(filtered_halfmap.astype(np.float32))
        #    output_mrc.voxel_size = voxel_size

        # replace the edge of the map with the origional map
        d = filtered_halfmap.shape[0]
        r = np.arange(d)-d//2
        [Z,Y,X] = np.meshgrid(r,r,r)
        index = np.round(np.sqrt(Z**2+Y**2+X**2))
        filtered_halfmap[index>(d//2)] = halfmap[index>(d//2)]

        pred = network.predict_map(filtered_halfmap, output_dir=output_dir, cube_size = cube_size, crop_size=predict_crop_size)
        diff_map = (pred - filtered_halfmap)
        out_map = diff_map + halfmap

        with mrcfile.new(f"{output_dir}/netinput_{output_base}_iter{iter_count}.mrc", overwrite=True) as output_mrc:
            output_mrc.set_data(filtered_halfmap.astype(np.float32))
            output_mrc.voxel_size = voxel_size
     
        with mrcfile.new(f"{output_dir}/netoutput_{output_base}_iter{iter_count}.mrc", overwrite=True) as output_mrc:
            output_mrc.set_data(pred.astype(np.float32))
            output_mrc.voxel_size = voxel_size
        
        with mrcfile.new(f"{output_dir}/corrected_{output_base}_iter{iter_count}.mrc", overwrite=True) as output_mrc:
            output_mrc.set_data(out_map.astype(np.float32))
            output_mrc.voxel_size = voxel_size

        files = os.listdir(output_dir)
        for item in files:
            if item == "data" or item == "data~":
                path = f'{output_dir}/{item}'
                shutil.rmtree(path)
            if item.startswith('subvolume'):
                path = f'{output_dir}/{item}'
                os.remove(path)      
        logging.info('Done predicting')
        
    
    
    
# def map_refine_multi(halfmap, mask, fsc3d, voxel_size, limit_res, output_dir = "results", output_base="half1", n_subvolume = 50, cube_size = 64, crop_size = 96, weighting=False, noise_scale=None):
#     log_level = "info"
#     if log_level == "debug":
#         logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
#         datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])
#     else:
#         logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
#         datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
#         #logging.basicConfig(format='%(asctime)s.%(msecs)03d, %(levelname)-8s %(message)s',
#         #datefmt="%Y-%m-%d,%H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])

#     n_sub_list = []
#     count = 0
#     for i,m in enumerate(mask):
#         n = np.count_nonzero(mask[i]>0.5)
#         count += n
#         n_sub_list.append(n)
#     logging.info(n_sub_list)
#     for i in range(len(n_sub_list)):
#         n_sub_list[i] = int(n_sub_list[i]/count * n_subvolume) + 1
#     logging.info(n_sub_list)

#     # Fixed parameters
#     num_iterations = 10
#     data_dir = output_dir+"/data"
#     mkfolder(data_dir)
#     fsc3d, fsc3d_cube, fsc3d_full = process_3dfsc(halfmap[0],fsc3d,weighting,crop_size,cube_size,voxel_size,limit_res)

#     from IsoNet.models.network import Net
#     network = Net(fsc3d=fsc3d_cube)
#     current_map = []
#     #main iterations
#     for iter_count in range(1,num_iterations+1):
#         mkfolder(data_dir)
#         for i,h in enumerate(halfmap):
#             previous_filename = "{}/corrected{}_{}_iter{}.mrc".format(output_dir, i, output_base, iter_count-1)
#             print(previous_filename)

#             if iter_count > 1:
#                 with mrcfile.open(previous_filename, 'r') as mrc:
#                     current_map[i] = mrc.data
#                 current_map[i] = normalize(current_map[i],percentile=False)
#             else:
#                 halfmap[i] = normalize(halfmap[i],percentile=False)
#                 current_map.append(halfmap[i].copy())
            
#                 with mrcfile.new(previous_filename, overwrite=True) as mrc:
#                     mrc.set_data(halfmap[i])
                
            
#             mrc_list = extract_subvolume(current_map[i], n_sub_list[i], crop_size, mask[i], output_dir,prefix=str(i))

#             #if noise_scale is not None:
#             #    noise_scale=np.std(current_map[mask>0.1])*0.2
#             noise_scale = 0
#             noise_mean = 0
#             logging.info("Start Iteration{}!".format(iter_count))

#             logging.info("Start preparing subvolumes!")
#             get_cubes_list(fsc3d, mrc_list, data_dir, output_dir, crop_size, cube_size, noise_scale = noise_scale, noise_mean = noise_mean,prefix=str(i))
#             logging.info("Done preparing subvolumes!")
#         split_train_test(data_dir,batch_size=8)

#         ### start training and save model and json ###
#         logging.info("Start training!")

#         metrics = network.train(data_dir,gpuID=0, batch_size=8, epochs = 10, steps_per_epoch = 200, acc_grad = False) #train based on init model and save new one as model_iter{num_iter}.h5
#         logging.info("Start predicting!")
#         #network.predict(mrc_list, result_dir, iter_count+1, mw3d=fsc3d)
#         #logging.info("Done predicting subvolumes!")

#         #logging.info("Done training!")
#         #current_filename_n = "{}/corrected_norm_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
#         #current_filename = "{}/corrected_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
#         #current_filename1 = "{}/corrected1_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
#         #current_filename2 = "{}/corrected2_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
#         #outData = network.predict_map(halfmap, fsc3d_full, fsc3d, output_file=current_filename, voxel_size = voxel_size )
#         network.save("{}/model_{}_iter{}.h5".format(output_dir, output_base, iter_count))
#         for i,h in enumerate(halfmap):
#             current_map[i] = fsc_filter(halfmap[i], fsc3d_full)
#             current_filename = "{}/corrected{}_{}_iter{}.mrc".format(output_dir,i, output_base, iter_count) 
#             current_filename1 = "{}/corrected1_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
#             current_filename2 = "{}/corrected2_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
#             network.predict_map(current_map[i],halfmap[i], fsc3d_full, fsc3d, output_file=current_filename, voxel_size = voxel_size )
#                 #outData = normalize(outData,percentile=args.normalize_percentile)
#         #with mrcfile.new(current_filename, overwrite=True) as output_mrc:
#         #    output_mrc.set_data(outData.astype(np.float32))
#         #    output_mrc.voxel_size = voxel_size
#         # with mrcfile.new(current_filename1, overwrite=True) as output_mrc:
#         #     output_mrc.set_data((outData+halfmap).astype(np.float32))
#         #     output_mrc.voxel_size = voxel_size
#         # with mrcfile.new(current_filename2, overwrite=True) as output_mrc:
#         #     output_mrc.set_data((outData+halfmap_origional).astype(np.float32))
#         #     output_mrc.voxel_size = voxel_size

#         logging.info('Done predicting')
#         #network.predict_map(normalize(halfmap), fsc3d, output_file=current_filename_n, voxel_size = voxel_size )
    
