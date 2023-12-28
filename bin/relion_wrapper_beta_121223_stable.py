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
from subprocess import check_output


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

def execute_combine(f1,f2,f3,limit_res=20, mask_file=None): 
    params = ' eval "$(conda shell.bash hook)" && conda activate %s && ' %CONDA_ENV     
    params += ' spisonet.py combine_map '
    params += ' %s' %(f1)  
    params += ' %s' %(f2)  
    params += ' %s' %(f3)  
    params += ' %s' %(limit_res)
    params += ' --mask_file %s' %(mask_file)
    print(params)
    os.system(params)   

def parse_env(ENV_STRING, val_type, default, silence=True):
    val = os.getenv(ENV_STRING)
    if val_type == "bool":
        if val=="True" or val=="true" or val=="TRUE" or val==True:
            val = True
        elif val=="False" or val=="false" or val=="FALSE" or val==False:
            val = False
        else:
            if not silence:
                print(f"{ENV_STRING}={val} does not match True or False")
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
            if not silence:
                print(f"{val} is None")
            val = default
    if not silence:
        print(f"set {ENV_STRING}={val}")
    return val

def parse_filename(star):
    dir=os.path.dirname(star)

    part = star.split('/')[-1].split('_')
    iter_string = part[1]
    basename = part[0]
    iter_number = int(iter_string[2:5])
    half_str = part[2]

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

def wait_until_file(sync_file,interval_time=10,total_time=10000):
    time.sleep(interval_time)
    for i in range (1,total_time):
        try:
            with open(sync_file, 'r') as f:
                result =  f.read()
            s = f'rm {sync_file}'
            check_output(s, shell=True)
            return result
        except:
            time.sleep(interval_time)

def write_to_file(file_name, words='This is a temperal file to synchronize half 1 and 2 for isonet'):
    with open(file_name, 'w') as f:
        f.write(words)



if __name__=="__main__":  
    dir, basename, half_str, var, beforeVar = parse_filename(sys.argv[1])
    if half_str == "half2":
        silence = True
    else:
        silence = False
        print("iter =", var)

    import torch
    gpu_list = list(range(torch.cuda.device_count()))
    gpu_list=','.join(map(str, gpu_list))

    gpu = parse_env("CUDA_VISIBLE_DEVICES", "string", gpu_list, silence)
    CONDA_ENV = parse_env("CONDA_ENV", "string", None, silence)
    whitening = parse_env("ISONET_WHITENING", "bool", True, silence)
    whitening_low = parse_env("ISONET_WHITENING_LOW", "float", 10, silence)
    retrain = parse_env("ISONET_RETRAIN_EACH_ITER", "bool", False, silence)
    beta = parse_env("ISONET_BETA", "float", 0.5, silence)
    alpha = parse_env("ISONET_ALPHA", "float", 1, silence)
    
    limit_healpix = parse_env("ISONET_START_HEALPIX", "int", 3, silence)
    acc_batches = parse_env("ISONET_ACC_BATCHES", "int", 2, silence)
    epochs = parse_env("ISONET_EPOCHS", "int", 5, silence)
    start_epochs = parse_env("ISONET_START_EPOCHS", "int", 5, silence)
    combine = parse_env("ISONET_COMBINE", "bool", False, silence)
    lowpass = parse_env("ISONET_LOWPASS", "bool", True, silence)

    # resolution_initial
    with open("%s/%s_it000_optimiser.star" %(dir,basename)) as file:
        for line_number,li in enumerate(file.readlines()):
            if '--combine' in li:
                combine = True
                break
    if not silence:
        print("set combine:",combine)       

    # healpix
    with open("%s/%s_it%s_sampling.star" %(dir,basename,beforeVar)) as file:
        for li in file.readlines():
            if "rlnHealpixOrder " in li: 
                healpix = int(li.split()[1]) 
                if not silence:
                    print("healpix = %s" %healpix)
                break

    # mask_file
    mask_file = None
    with open("%s/%s_it%s_optimiser.star" %(dir,basename,beforeVar)) as file:
        for li in file.readlines():
            if "_rlnSolventMaskName " in li: 
                mask_file = li.split()[1]
                if not silence:
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
                if not silence:
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

        if float(limit_resolution) > resolution_initial:
            limit_resolution = resolution_initial
        if not silence:
            print(f"real limit resolution to {limit_resolution}")

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
            execute_whitening(mrc_unfil, mrc_unfil, mask_file, high_res=limit_resolution, low_res=whitening_low)  
            shutil.copy(mrc_unfil, mrc_whiten_backup)
            shutil.copy(mrc_unfil, mrc_overwrite)

        if combine:
            execute_combine(mrc_initial,mrc_unfil,mrc_unfil,resolution_initial, mask_file=mask_file) 
            shutil.copy(mrc_unfil, mrc_combine_backup)
            shutil.copy(mrc_unfil, mrc_overwrite)

        if lowpass:
            s = f"relion_image_handler --i {mrc_unfil} --o {mrc_lowpass_backup} --lowpass {limit_resolution}; cp {mrc_lowpass_backup} {mrc_unfil}"
            print(s)
            check_output(s, shell=True)
            shutil.copy(mrc_unfil, mrc_overwrite) 

        # remove tempory
        remove_intermediate = True
        if remove_intermediate:
            try:
                os.remove('%s/%s_it%s_%s_class001_external_reconstruct_data_real.mrc' %(dir,basename,var,half_str))
                os.remove('%s/%s_it%s_%s_class001_external_reconstruct_data_imag.mrc' %(dir,basename,var,half_str)) 
                os.remove('%s/%s_it%s_%s_class001_external_reconstruct_weight.mrc' %(dir,basename,var,half_str)) 
            except:
                print("do not remove tempory files")

        sync3 = '%s/%s_it%s_class001_external_reconstruct.sync3' %(dir,basename,var)
        sync4 = '%s/%s_it%s_class001_external_reconstruct.sync4' %(dir,basename,var)

        if (healpix >= limit_healpix) and (limit_resolution < 15) and (half_str == "half2"):
            write_to_file(sync4)
            wait_until_file(sync3)

        elif (healpix >= limit_healpix) and (limit_resolution < 15) and (half_str == "half1"):
            wait_until_file(sync4)
            
            mrc1_overwrite = '%s/%s_it%s_half1_class001_external_reconstruct.mrc' %(dir,basename,var)
            mrc2_overwrite = '%s/%s_it%s_half2_class001_external_reconstruct.mrc' %(dir,basename,var)
            mrc1_unfil = '%s/%s_it%s_half1_class001_unfil.mrc' %(dir,basename,var)
            mrc2_unfil = '%s/%s_it%s_half2_class001_unfil.mrc' %(dir,basename,var)
            mrc1_cor = '%s/corrected_%s_it%s_half1_class001_unfil.mrc' %(dir,basename,var)
            mrc2_cor = '%s/corrected_%s_it%s_half2_class001_unfil.mrc' %(dir,basename,var)
            fscn='%s/%s_it%s_3DFSC.mrc' %(dir,basename,var)    

            #Force write pixelsize
            with mrcfile.open(mrc1_unfil) as f1:
                emMap1 = f1.data.astype(np.float32).copy()  
            with mrcfile.open(mrc2_unfil) as f2:
                emMap2 = f2.data.astype(np.float32).copy()  

            with mrcfile.new(mrc1_unfil, overwrite=True) as f1:
                f1.set_data(emMap1)             
                f1.voxel_size = tuple([sampling]*3)
            with mrcfile.new(mrc2_unfil, overwrite=True) as f2:
                f2.set_data(emMap2) 
                f2.voxel_size = tuple([sampling]*3) 

            # execute 3dfsc
            execute_3dfsc(mrc1_unfil, mrc2_unfil, fscn, limit_res=limit_resolution, mask_file=mask_file)    
            print(f"using FSC3D file {fscn}")

            mean1_before =  emMap1.mean()                  
            mean2_before =  emMap2.mean()  
            std1_before =  emMap1.std()                  
            std2_before =  emMap2.std()  
        
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

            # lowpass again
            if lowpass:
                from subprocess import check_output
                s = f"relion_image_handler --i {mrc1_overwrite} --o tmp.mrc --lowpass {limit_resolution}; mv tmp.mrc {mrc1_overwrite}"
                check_output(s, shell=True)
                s = f"relion_image_handler --i {mrc2_overwrite} --o tmp.mrc --lowpass {limit_resolution}; mv tmp.mrc {mrc2_overwrite}"
                check_output(s, shell=True)

            print("finished spisonet reconstruction")
            write_to_file(sync3)




        
            
