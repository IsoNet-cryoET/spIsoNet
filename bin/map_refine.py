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


def crop_to_size(array, crop_size, cube_size):
        start = crop_size//2 - cube_size//2
        end = crop_size//2 + cube_size//2
        return array[start:end,start:end,start:end]
def fsc_filter(map,fsc3d):
    #crop_size = map.shape[0]
    mw = fsc3d
    # mw = skimage.transform.resize(fsc3d, [crop_size,crop_size,crop_size], order = 0)

    # half_size = crop_size // 2 
    # if limit_r1 <1 and limit_r1>0:
    #     for i in range(crop_size):
    #         for j in range(crop_size):
    #             for k in range(crop_size):
    #                 r = ((i-half_size)**2 + (j-half_size)**2 + (k-half_size)**2)**0.5
    #                 if r > limit_r1 * half_size and r < limit_r2 * half_size:
    #                     mw[i,j,k] = 1

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



def rescale_fsc(fsc3d, limit_r1, limit_r2, crop_size):
    fsc3d = skimage.transform.resize(fsc3d, [crop_size,crop_size,crop_size], order = 0)
    fsc3d[fsc3d<0] = 0
    half_size = crop_size // 2 
    print("lr2",limit_r2)
    print("lr1",limit_r1)

    for i in range(crop_size):
        for j in range(crop_size):
            for k in range(crop_size):
                r = ((i-half_size)**2 + (j-half_size)**2 + (k-half_size)**2)**0.5
                if r > limit_r1 * half_size or r < limit_r2 * half_size:
                    fsc3d[i,j,k] = 1
                #if r < 0.7* limit_r * half_size:
                #    fsc3d[i,j,k] = 1
                #d = r-(limit_r*half_size) #30
                #fsc3d[i,j,k]=max(1/(1+np.exp(-d)), fsc3d[i,j,k])
    fsc3d[half_size,half_size,half_size]=1
    #fsc3d[fsc3d>0.143] =1
    
    #fsc3d[fsc3d<=0.143] =0
    return fsc3d


def get_cubes(mw3d, data_dir, output_dir, crop_size, cube_size, noise_scale,noise_mean,prefix, inp):
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
    noise_a = apply_wedge_dcube(noise_a, mw3d=mw3d, ld1 = 0, ld2 = 1)

    for i in range(len(rotation_list)): 
        #data_mask = crop_to_size(rotated_mask[i], crop_size, cube_size)

        data_X = crop_to_size(datax[i], crop_size, cube_size)
        data_Y = crop_to_size(rotated_data[i], crop_size, cube_size)
        noise = crop_to_size(noise_a[i], crop_size, cube_size)

        #data_X = data_X*data_mask + data_Y*(1-data_mask) + noise* noise_scale
        data_X = data_X + noise* noise_scale
        data_Y = data_Y - data_X
        #print(np.average(data_Y[data_mask<0.5]))
        #data_Y = data_Y*data_mask + noise_mean * (1-data_mask)# + data_Y*(1-data_mask)# + 0.5*data_Y*(1-data_mask)
        # with mrcfile.new('{}/train_x/x_{}.mrc'.format(data_dir, start), overwrite=True) as output_mrc:
        #     output_mrc.set_data(data_X.astype(np.float32))
        # with mrcfile.new('{}/train_y/y_{}.mrc'.format(data_dir, start), overwrite=True) as output_mrc:
        #     output_mrc.set_data(data_Y.astype(np.float32))
        # start += 1
        with mrcfile.new('{}/train_x/x{}_{}.mrc'.format(data_dir,prefix, start), overwrite=True) as output_mrc:
            output_mrc.set_data(data_X.astype(np.float32))
        with mrcfile.new('{}/train_y/y{}_{}.mrc'.format(data_dir,prefix, start), overwrite=True) as output_mrc:
            output_mrc.set_data(data_Y.astype(np.float32))
        start += 1


def get_cubes_list(mw3d, mrc_list, data_dir,output_dir, crop_size, cube_size, batch_size=8, noise_scale = 1, noise_mean = 0, prefix=''):
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

    preprocessing_ncpus = 12
    if preprocessing_ncpus > 1:
        func = partial(get_cubes, mw3d, data_dir, output_dir, crop_size, cube_size,noise_scale, noise_mean, prefix)
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

def process_3dfsc(halfmap,fsc3d,weighting,crop_size,cube_size,voxel_size,limit_res):
    # Fixed parameters
    full_size = halfmap.shape[0]

    fsc3d = fsc3d.copy()


    limit_r = (2*voxel_size)/limit_res
    limit_r2 = 0 #(2*voxel_size)/10
 
    if weighting:
        #logging.info("Apply sqrt(2*FSC/(1+FSC)) for 3DFSC")
        #fsc3d = np.sqrt((2*fsc3d)/(1+fsc3d))
        logging.info("Apply sqrt(FSC) for 3DFSC")
        fsc3d = np.sqrt(fsc3d)
    else:
        logging.info("No sqrt(2*FSC/(1+FSC)) for 3DFSC")

    fsc3d = rescale_fsc(fsc3d, limit_r, limit_r2, full_size)
    fsc3d_full = fsc3d.copy()
    with mrcfile.new("fouriermask.mrc",overwrite=True) as mrc:
        mrc.set_data(fsc3d)
    # with mrcfile.new("psf.mrc",overwrite=True) as mrc:
    #     mrc.set_data(np.real(np.fft.ifftn(np.fft.fftshift(fsc3d))).astype(np.float32))

    fsc3d = rescale_fsc(fsc3d_full, limit_r, limit_r2, crop_size)
    fsc3d_cube = rescale_fsc(fsc3d_full, limit_r, limit_r2, cube_size)
    threshold = 0.7
    fsc3d[fsc3d<threshold] = 0
    fsc3d[fsc3d>threshold] = 1
    fsc3d_cube[fsc3d_cube<threshold] = 0
    fsc3d_cube[fsc3d_cube>threshold] = 1
    fsc3d_full[fsc3d_full<threshold] = 0
    fsc3d_full[fsc3d_full>threshold] = 1
    return fsc3d,fsc3d_cube, fsc3d_full

def map_refine(halfmap, mask, fsc3d, voxel_size, limit_res, output_dir = "results", output_base="half1", n_subvolume = 50, cube_size = 64, crop_size = 96, weighting=False, noise_scale=None):
    log_level = "info"
    if log_level == "debug":
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])
    else:
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
        datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
        #logging.basicConfig(format='%(asctime)s.%(msecs)03d, %(levelname)-8s %(message)s',
        #datefmt="%Y-%m-%d,%H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])

    num_iterations = 9
    data_dir = output_dir+"/data"
    mkfolder(data_dir)
    fsc3d, fsc3d_cube, fsc3d_full = process_3dfsc(halfmap,fsc3d,weighting,crop_size,cube_size,voxel_size,limit_res)

    from IsoNet.models.network import Net
    network = Net(fsc3d=fsc3d_cube)

    halfmap = normalize(halfmap)#, pmin=0, pmax=100)
    current_map = halfmap.copy()
    #noise_scale=np.std(current_map[mask<0.1])*1
    if noise_scale is not None:
        noise_scale=np.std(current_map[mask>0.1])*0.2
    noise_mean=np.average(current_map[mask<0.1])
    
    #main iterations
    for iter_count in range(0,num_iterations+1):
        
        if iter_count == 0:
            current_filename = "{}/corrected_{}_iter{}.mrc".format(output_dir, output_base, iter_count)
            with mrcfile.new(current_filename, overwrite=True) as mrc:
                mrc.set_data(current_map)
                mrc.update_header_from_data()
                mrc._set_voxel_size(voxel_size,voxel_size,voxel_size)
            continue
        
        previous_filename = "{}/corrected_{}_iter{}.mrc".format(output_dir, output_base, iter_count-1)
            
        with mrcfile.open(previous_filename, 'r') as mrc:
            current_map = mrc.data
           
        #current_map = normalize(current_map)

        #extract subvolume
        # seeds=create_cube_seeds(current_map, n_subvolume, crop_size, mask)
        # subtomos=crop_cubes(current_map,seeds,crop_size)
        # submasks=crop_cubes(mask,seeds, crop_size)
        # mrc_list = []
        # for j,s in enumerate(subtomos):
        #     im_name = '{}/subvolume_{:0>6d}.mrc'.format(output_dir, j)
        #     with mrcfile.new(im_name, overwrite=True) as output_mrc:
        #         output_mrc.set_data(s.astype(np.float32))
        #     msk_name = '{}/mask_{:0>6d}.mrc'.format(output_dir, j)
        #     with mrcfile.new(msk_name, overwrite=True) as output_mrc:
        #         output_mrc.set_data(submasks[j].astype(np.float32))
        #     mrc_list.append(im_name)
        mrc_list = extract_subvolume(current_map, n_subvolume, crop_size, mask, output_dir)

        logging.info("Start Iteration{}!".format(iter_count))

        logging.info("Start preparing subvolumes!")
        #get_cubes_list(fsc3d, mrc_list, data_dir, output_dir, crop_size, cube_size, batch_size=8, noise_scale = noise_scale, noise_mean = noise_mean)
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
        #current_map = fsc_filter(current_map, fsc3d_full)
        #filtered_halfmap = halfmap#fsc_filter(halfmap, fsc3d_full)
        filtered_halfmap = fsc_filter(halfmap, fsc3d_full)
        current_filename = "{}/corrected_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
        output_sigma_file = "{}/sigma_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
        current_filename1 = "{}/corrected1_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
        current_filename2 = "{}/corrected2_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
        
        print("\nvoxelsizeiter0:", voxel_size)
        #outData = network.predict_map_sigma(current_map,halfmap, fsc3d_full, fsc3d, output_file=current_filename, voxel_size = voxel_size, output_sigma_file=output_sigma_file )
        outData = network.predict_map(filtered_halfmap,halfmap, fsc3d_full, fsc3d, output_file=current_filename, voxel_size = voxel_size)

        print("\nvoxelsizeiter0:", voxel_size)
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

    
    
    
def map_refine_multi(halfmap, mask, fsc3d, voxel_size, limit_res, output_dir = "results", output_base="half1", n_subvolume = 50, cube_size = 64, crop_size = 96, weighting=False, noise_scale=None):
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
    num_iterations = 10
    data_dir = output_dir+"/data"
    mkfolder(data_dir)
    fsc3d, fsc3d_cube, fsc3d_full = process_3dfsc(halfmap[0],fsc3d,weighting,crop_size,cube_size,voxel_size,limit_res)

    from IsoNet.models.network import Net
    network = Net(fsc3d=fsc3d_cube)

    current_map = []
    #main iterations
    for iter_count in range(1,num_iterations+1):
        mkfolder(data_dir)
        for i,h in enumerate(halfmap):
            previous_filename = "{}/corrected{}_{}_iter{}.mrc".format(output_dir, i, output_base, iter_count-1)
            print(previous_filename)

            if iter_count > 1:
                with mrcfile.open(previous_filename, 'r') as mrc:
                    current_map[i] = mrc.data
                current_map[i] = normalize(current_map[i])
            else:
                halfmap[i] = normalize(halfmap[i])
                current_map.append(halfmap[i].copy())
            
                with mrcfile.new(previous_filename, overwrite=True) as mrc:
                    mrc.set_data(halfmap[i])
                
            
            mrc_list = extract_subvolume(current_map[i], n_sub_list[i], crop_size, mask[i], output_dir,prefix=str(i))

            #if noise_scale is not None:
            #    noise_scale=np.std(current_map[mask>0.1])*0.2
            noise_scale = 0
            noise_mean = 0
            logging.info("Start Iteration{}!".format(iter_count))

            logging.info("Start preparing subvolumes!")
            get_cubes_list(fsc3d, mrc_list, data_dir, output_dir, crop_size, cube_size, noise_scale = noise_scale, noise_mean = noise_mean,prefix=str(i))
            logging.info("Done preparing subvolumes!")
        split_train_test(data_dir,batch_size=8)

        ### start training and save model and json ###
        logging.info("Start training!")

        metrics = network.train(data_dir,gpuID=0, batch_size=8, epochs = 10, steps_per_epoch = 200, acc_grad = False) #train based on init model and save new one as model_iter{num_iter}.h5
        logging.info("Start predicting!")
        #network.predict(mrc_list, result_dir, iter_count+1, mw3d=fsc3d)
        #logging.info("Done predicting subvolumes!")

        #logging.info("Done training!")
        #current_filename_n = "{}/corrected_norm_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
        #current_filename = "{}/corrected_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
        #current_filename1 = "{}/corrected1_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
        #current_filename2 = "{}/corrected2_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
        #outData = network.predict_map(halfmap, fsc3d_full, fsc3d, output_file=current_filename, voxel_size = voxel_size )
        network.save("{}/model_{}_iter{}.h5".format(output_dir, output_base, iter_count))
        for i,h in enumerate(halfmap):
            current_map[i] = fsc_filter(halfmap[i], fsc3d_full)
            current_filename = "{}/corrected{}_{}_iter{}.mrc".format(output_dir,i, output_base, iter_count) 
            current_filename1 = "{}/corrected1_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
            current_filename2 = "{}/corrected2_{}_iter{}.mrc".format(output_dir, output_base, iter_count) 
            network.predict_map(current_map[i],halfmap[i], fsc3d_full, fsc3d, output_file=current_filename, voxel_size = voxel_size )
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
    
