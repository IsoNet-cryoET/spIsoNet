# -*- coding: utf-8 -*-


# To execute this script in relion_refine it is necessary to use the argument "--external_reconstruct".
# It is also required to set the RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE environment variable 
# to point to this script.


#export RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE="python /home/cii/software/spIsoNet/bin/relion_wrapper.py"
#export CONDA_ENV="torch2-test"
#export CUDA_VISIBLE_DEVICES="0"



import os
import os.path
import sys
import time
import numpy as np
import mrcfile
import shutil


#for isonet 
'''
need GPU information 
'''

def execute_external_relion(star):

    params = ' relion_external_reconstruct'
    params += ' %s' %star
    os.system(params)     
    
def execute_deep(mrc1, mrc2, fsc3d, dir, gpu, epochs = 1, mask_file = None, pretrained_model=None, alpha = None, acc_batches= None, batch_size = None, beta=None): 
    #data_file =  ' %s/%s_it%s_half%s_class001_external_reconstruct.mrc' %(dir, basename, var, half)   
    params = ' eval "$(conda shell.bash hook)" && conda activate %s && ' %CONDA_ENV     
    params += ' spisonet.py refine_n2n '
    params += f" {mrc1} {mrc2}"      
    params += ' %s' %(fsc3d) 
    params += ' --epochs %s --n_subvolume 1000'   %(epochs)
    if acc_batches is not None:
        params += ' --acc_batches %s'   %(acc_batches) 
    if batch_size is not None:
        params += ' --batch_size %s'   %(batch_size) 
    if alpha is not None:
        params += ' --alpha %s'   %(alpha) 
    if beta is not None:
        params += ' --beta %s'   %(beta) 
    params += ' --output_dir %s' %(dir) 
    params += ' --gpuID %s' %(gpu) 
    if pretrained_model is not None:
        params += ' --pretrained_model %s' %(pretrained_model)
    if mask_file is not None:
        params += ' --mask %s' %(mask_file)

    print(params)
    os.system(params)

def execute_3dfsc(fn1,fn2,fscn,limit_res=None, mask_file=None): 
    params = ' eval "$(conda shell.bash hook)" && conda activate %s && ' %CONDA_ENV     
    params += ' spisonet.py fsc3d '
    params += ' %s' %(fn1)  
    params += ' %s' %(fn2) 
    params += ' -o %s' %(fscn)
    if limit_res is not None:
        params += ' --limit_res %s'%(limit_res)
    if mask_file is not None:
        params += ' --mask %s'%(mask_file)
#     params += ' -p wideTarget' 
    os.system(params)

def execute_whitening(fn1,fscn,mask,high_res,low_res=10): 
    params = ' eval "$(conda shell.bash hook)" && conda activate %s && ' %CONDA_ENV     
    params += ' spisonet.py whitening '
    params += ' %s' %(fn1)  
    params += ' -o %s' %(fscn)
    params += ' --mask %s' %(mask)
    params += ' --high_res %s'%(high_res)
    params += ' --low_res %s'%(low_res)
    print(params)
    os.system(params)

def execute_combine(f1,f2,f3,limit_res=20): 
    params = ' eval "$(conda shell.bash hook)" && conda activate %s && ' %CONDA_ENV     
    params += ' spisonet.py combine_map '
    params += ' %s' %(f1)  
    params += ' %s' %(f2)  
    params += ' %s' %(f3)  
    params += ' %s' %(limit_res)
    print(params)
    os.system(params)   

def parse_env(ENV_STRING, val_type, default):
    val = os.getenv(ENV_STRING)
    if val_type == "bool":
        if val=="True" or val=="true" or val=="TRUE" or val==True:
            val = True
        elif val=="False" or val=="false" or val=="FALSE" or val==False:
            val = False
        else:
            print(f"{ENV_STRING}={val} does match True or False")
            val = default

    elif val_type == 'int':
        if val is None or len(val) == 0:
            val = default
        else:
            val = int(val)

    elif val_type == 'float':
        if val is None or len(val) == 0:
            val = default
        else:
            val = float(val)
    
    elif val_type == 'str':
        if val is None or len(val) == 0:
            print(f"{val} is None")
            val = default
    
    print(f"set {ENV_STRING}={val}")
    return val

def parse_filename(star):
    dir=os.path.dirname(star)

    part = star.split('/')[-1].split('_')
    iter_string = part[1]
    basename = part[0]
    iter_number = int(iter_string[2:5])
    print("iter =", iter_number)
    half_str = part[2]
    print('half_str',half_str)

    if int(iter_number) <= 9: 
        var = '00%d' %(iter_number)
        beforeVar = '00%d' %(iter_number-1)
    elif int(iter_number) == 10:
        var = '0%d' %(iter_number)
        beforeVar = '00%d' %(iter_number-1)
    else:
        var = '0%d' %(iter_number)
        beforeVar = '0%d' %(iter_number-1)
    return dir, basename, half_str, var, beforeVar

if __name__=="__main__":  

    dir, basename, half_str, var, beforeVar = parse_filename(sys.argv[1])
   
    import torch
    gpu_list = list(range(torch.cuda.device_count()))
    gpu_list=','.join(map(str, gpu_list))

    gpu = parse_env("CUDA_VISIBLE_DEVICES", "string", gpu_list)
    CONDA_ENV = parse_env("CONDA_ENV", "string", None)
    whitening = parse_env("ISONET_WHITENING", "bool", False)
    whitening_low = parse_env("ISONET_WHITENING_LOW", "float", 10)
    retrain = parse_env("ISONET_RETRAIN_EACH_ITER", "bool", False)
    beta = parse_env("ISONET_BETA", "float", 0.5)
    alpha = parse_env("ISONET_ALPHA", "float", 1)
    
    limit_healpix = parse_env("ISONET_START_HEALPIX", "int", 3)
    acc_batches = parse_env("ISONET_ACC_BATCHES", "int", 2)
    epochs = parse_env("ISONET_EPOCHS", "int", 5)
    start_epochs = parse_env("ISONET_START_EPOCHS", "int", 5)
    combine = parse_env("ISONET_COMBINE", "bool", True)
    lowpass = parse_env("ISONET_LOWPASS", "bool", True)

    # resolution_initial
    with open("%s/%s_it000_optimiser.star" %(dir,basename)) as file:
        for line_number,li in enumerate(file.readlines()):
            if '--combine' in li:
                combine = True
                break
    print("combine:",combine)            
    # healpix
    with open("%s/%s_it%s_sampling.star" %(dir,basename,beforeVar)) as file:
        for li in file.readlines():
            if "rlnHealpixOrder " in li: 
                healpix = int(li.split()[1]) 
                print("healpix = %s" %healpix)
                break

    # mask_file
    mask_file = None
    with open("%s/%s_it%s_optimiser.star" %(dir,basename,beforeVar)) as file:
        for li in file.readlines():
            if "_rlnSolventMaskName " in li: 
                mask_file = li.split()[1]
                print("mask_file = %s" %mask_file)
                break
        sampling_index = None

    # sampling(pixelsize)        
    with open("%s/%s_it%s_data.star" %(dir,basename,beforeVar)) as f:
        for line in f.readlines():
            if "_rlnImagePixelSize" in line:
                sampling_index = int(line.split()[1].split("#")[1])
            if "opticsGroup1" in line:
                sampling = float(line.split()[sampling_index-1])
                print("pixel size = %s" %sampling) 

    # limit_resolution: last iter resolution at 0.143
    limit_resolution = 2*sampling
    start_check = 10000
    with open("%s/%s_it%s_half1_model.star" %(dir,basename,beforeVar)) as file:
        for line_number,li in enumerate(file.readlines()):
            if "_rlnEstimatedResolution " in li:
                resolution_index = int(li.split()[1].split("#")[1])
            if "_class001.mrc" in li:
                resolution = float(li.split()[resolution_index-1])
            if "_rlnAngstromResolution" in li:
                Aresolution_index = int(li.split()[1].split("#")[1])
            if "_rlnGoldStandardFsc" in li:
                FSC_index = int(li.split()[1].split("#")[1])
                start_check = line_number
            if line_number >= start_check:
                line_split = li.split()
                if len(line_split)>FSC_index:
                    if float(line_split[FSC_index-1])<= 0.143:
                        limit_resolution = float(line_split[Aresolution_index-1])
                        break
    # resolution_initial
    with open("%s/%s_it000_optimiser.star" %(dir,basename)) as file:
        for line_number,li in enumerate(file.readlines()):
            if 'ini_high' in li:
                li_sp = li.split()
                index_ini = li_sp.index("--ini_high")
                resolution_initial = float(li_sp[index_ini+1])
                break

    check_final = (half_str == "class001") 

    execute_external_relion(sys.argv[1]) 

    if (check_final is True):
        mrc_final1 = '%s/%s_half1_class001_unfil.mrc' %(dir,basename)
        mrc_final2 = '%s/%s_half2_class001_unfil.mrc' %(dir,basename)
        final_fsc = '%s/%s_3DFSC.mrc' %(dir,basename)
        print("-----------")
        print("Important information for the final iteration!!!")
        print("The final half_unfil maps are generated with relion not spIsoNet")
        print("You may want to further use spIsoNet to improve the final maps")
        print("spIsoNet commands to correct final maps are:")

    else:
        print('Resolution in previous iteration', resolution)
        print('limit resolution to FSC=0.143', limit_resolution)
        mrc_initial = '%s/%s_it000_%s_class001.mrc' %(dir,basename,half_str)
        mrc_unfil = '%s/%s_it%s_%s_class001_unfil.mrc' %(dir,basename,var,half_str)
        mrc_unfil_backup = '%s/%s_it%s_%s_class001_unfil_backup.mrc' %(dir,basename,var,half_str)
        mrc_lowpass_backup = '%s/%s_it%s_%s_class001_unfil_lowpass_backup.mrc' %(dir,basename,var,half_str)
        mrc_whiten_backup = '%s/%s_it%s_%s_class001_unfil_whiten_backup.mrc' %(dir,basename,var,half_str)
        mrc_combine_backup = '%s/%s_it%s_%s_class001_unfil_combine_backup.mrc' %(dir,basename,var,half_str)
        mrc_overwrite = '%s/%s_it%s_%s_class001_external_reconstruct.mrc' %(dir,basename,var,half_str)
        mrc_overwrite_backup = '%s/%s_it%s_%s_class001_external_reconstruct_backup.mrc' %(dir,basename,var,half_str)
        shutil.copy(mrc_unfil, mrc_unfil_backup)
        shutil.copy(mrc_overwrite, mrc_overwrite_backup)
        shutil.copy(mrc_unfil, mrc_overwrite) 

        if limit_resolution < whitening_low and whitening:
            print("whitening")
            execute_whitening(mrc_unfil, mrc_unfil, mask_file, high_res=limit_resolution, low_res=whitening_low)  
            shutil.copy(mrc_unfil, mrc_whiten_backup)
            shutil.copy(mrc_unfil, mrc_overwrite)

        if combine:
            print("combining")
            execute_combine(mrc_initial,mrc_unfil,mrc_unfil,resolution_initial) 
            shutil.copy(mrc_unfil, mrc_combine_backup)
            shutil.copy(mrc_unfil, mrc_overwrite)
        print(f'here {lowpass}')
        if lowpass and limit_resolution < resolution_initial:
            print(mrc_unfil)
            print(limit_resolution)
            #with mrcfile.open(fscn, 'r') as mrc:
            #    fsc = mrc.data
            #from spIsoNet.util.FSC import recommended_resolution
            #res = recommended_resolution(fsc, sampling, 0.143)
            #print(f"3DFSC resolution {res}")
            from subprocess import check_output
            s = f"relion_image_handler --i {mrc_unfil} --o {mrc_lowpass_backup} --lowpass {limit_resolution}; mv {mrc_lowpass_backup} {mrc_unfil}"
            check_output(s, shell=True)
            shutil.copy(mrc_unfil, mrc_overwrite) 

        print("sync {}".format(half_str))

        if (healpix >= limit_healpix) and (limit_resolution < 10) and (half_str == 'half1'):

            mrc1_overwrite = '%s/%s_it%s_half1_class001_external_reconstruct.mrc' %(dir,basename,var)
            mrc2_overwrite = '%s/%s_it%s_half2_class001_external_reconstruct.mrc' %(dir,basename,var)
            mrc1_unfil = '%s/%s_it%s_half1_class001_unfil.mrc' %(dir,basename,var)
            mrc2_unfil = '%s/%s_it%s_half2_class001_unfil.mrc' %(dir,basename,var)
            mrc1_cor = '%s/corrected_%s_it%s_half1_class001_unfil.mrc' %(dir,basename,var)
            mrc2_cor = '%s/corrected_%s_it%s_half2_class001_unfil.mrc' %(dir,basename,var)

            time.sleep(30)
            for i in range (1,15):
                try:
                    with mrcfile.open(mrc2_unfil) as f2:
                        break
                    with mrcfile.open(mrc2_overwrite) as f2:
                        break
                except:
                    print("Waiting for half2")
                    time.sleep(10)

            #Force write pixelsize
            with mrcfile.open(mrc1_unfil) as f1:
                emMap1 = f1.data.astype(np.float32).copy()  
            with mrcfile.open(mrc2_unfil) as f2:
                emMap2 = f2.data.astype(np.float32).copy()   
            with mrcfile.new(mrc1_unfil, overwrite=True) as f1:
                f1.set_data(emMap1.astype(np.float32))
                f1.voxel_size = tuple([sampling]*3)
            with mrcfile.new(mrc2_unfil, overwrite=True) as f2:
                f2.set_data(emMap2.astype(np.float32))
                f2.voxel_size = tuple([sampling]*3)

            # remember mean and std
            #print("Whether whitened map is in correct absolute gray scale?")
            mean1_before =  emMap1.mean()                  
            mean2_before =  emMap2.mean()  
            std1_before =  emMap1.std()                  
            std2_before =  emMap2.std()  

            # execute 3dfsc
            fscn='%s/%s_it%s_3DFSC.mrc' %(dir,basename,var)    
            execute_3dfsc(mrc1_unfil,mrc2_unfil,fscn, limit_res=limit_resolution, mask_file=mask_file)    
            print(f"using FSC3D file {fscn}")
        
            # looking for pretrained model
            model = '%s/%s_it%s_half_class001_unfil.pt' %(dir,basename,beforeVar)
            print(model)
            if not os.path.isfile(model):
                print(f"first isonet reconstruction for {start_epochs} epochs, because previous iteration healpix order become {limit_healpix}")
                model = None
                epochs = start_epochs
            if retrain:
                print(f"retrain network each relion iteration")
                model = None
            if model is not None:
                print("reuse network from previous relion iteration")

            # Use previous model for prediction
            # if debug_mode:
            #     execute_deep(mrc1,fscn, dir, gpu, epochs = 0, mask_file = mask_file, pretrained_model = model, batch_size = batch_size, acc_batches=acc_batches, alpha=alpha, beta=beta)
            #     execute_deep(mrc2,fscn, dir,  gpu, epochs = 0, mask_file = mask_file, pretrained_model = model, batch_size = batch_size, acc_batches=acc_batches, alpha=alpha, beta=beta)
            #     shutil.move(out_mrc1, '%s/prepredicted_%s_it%s_half1_class001_%s.mrc' %(dir,basename,var,ext))
            #     shutil.move(out_mrc2, '%s/prepredicted_%s_it%s_half2_class001_%s.mrc' %(dir,basename,var,ext))

            # train and predict
            execute_deep(mrc1_unfil, mrc2_unfil, fscn, dir,  gpu, epochs = epochs, mask_file = mask_file, pretrained_model = model, batch_size = None, acc_batches=acc_batches, alpha=alpha, beta=beta)

            with mrcfile.open(mrc1_cor) as d1:
                emDeep1 = d1.data.astype(np.float32).copy() 
            with mrcfile.open(mrc2_cor) as d2:
                emDeep2 = d2.data.astype(np.float32).copy()              

            finalMap1 = emDeep1*float(std1_before)+mean1_before
            finalMap2 = emDeep2*float(std2_before)+mean2_before
            
            #save mrcfile
            with mrcfile.new(mrc1_overwrite, overwrite=True) as fMap1:
                fMap1.set_data(finalMap1.astype(np.float32))
                fMap1.voxel_size = tuple([sampling]*3)
            with mrcfile.new(mrc2_overwrite, overwrite=True) as fMap2:
                fMap2.set_data(finalMap2.astype(np.float32))
                fMap2.voxel_size = tuple([sampling]*3)

            # if not debug_mode:
            #     os.remove(out_mrc1)
            #     os.remove(out_mrc2)  

            if lowpass:
                #with mrcfile.open(fscn, 'r') as mrc:
                #    fsc = mrc.data
                #from spIsoNet.util.FSC import recommended_resolution
                #res = recommended_resolution(fsc, sampling, 0.143)
                #print(f"3DFSC resolution {res}")
                from subprocess import check_output
                s = f"relion_image_handler --i {mrc1_overwrite} --o tmp.mrc --lowpass {limit_resolution}; mv tmp.mrc {mrc1_overwrite}"
                check_output(s, shell=True)
                s = f"relion_image_handler --i {mrc2_overwrite} --o tmp.mrc --lowpass {limit_resolution}; mv tmp.mrc {mrc2_overwrite}"
                check_output(s, shell=True)

            isonet_done = '%s/%s_it%s_class001_external_reconstruct.done' %(dir,basename,var)
            with open(isonet_done, 'w') as f:
                f.write("spisonet finished for this iteration")

            print("finished spisonet reconstruction")
        if (healpix >= limit_healpix) and (limit_resolution < 10) and (half_str == 'half2'):
            isonet_done = '%s/%s_it%s_class001_external_reconstruct.done' %(dir,basename,var)
            time.sleep(30)
            for i in range (1,10000):
                try:
                    with open(isonet_done, 'r') as f2:
                        break
                except:
                    #print("Waiting")
                    time.sleep(30)

        
            
