# -*- coding: utf-8 -*-

#Wrapper to run relion_external_reconstruct and deepEMhancer
#Author: Erney Ramirez-Aportela
#contact: erney.ramirez@gmail.com

#Usage:
# To execute this script in relion_refine it is necessary to use the argument "--external_reconstruct".
# It is also required to set the RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE environment variable 
# to point to this script.

# Sometimes it is also necessary to set dynamic GPU allocation
# using the environment variable TF_FORCE_GPU_ALLOW_GROWTH='true'.

# For example:
# export RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE="python /path/to/relion_deepEMhancer_extRec.py"

# In order to run deepEMhancer it is necessary to activate 
# the conda environment used in deepEMhancer installation. 


#export RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE="python /home/cii/software/spIsoNet/bin/relion_wrapper.py"
#export CONDA_ENV="torch2-test"
#export CUDA_VISIBLE_DEVICES="0"



import os
import os.path
import sys
import time
import numpy as np
import mrcfile
# import fcntl
# import errno

#for isonet 
'''
need GPU information 
'''

def execute_external_relion(star):

    params = ' relion_external_reconstruct'
    params += ' %s' %star
    os.system(params)     
    
def execute_deep(data_file, dir, basename, var, gpu, epochs = 1, mask_file = None, pretrained_model=None): 
    print(CONDA_ENV) 
    #data_file =  ' %s/%s_it%s_half%s_class001_external_reconstruct.mrc' %(dir, basename, var, half)   
    print(f"processing {data_file}")  
    params = ' eval "$(conda shell.bash hook)" && conda activate %s && ' %CONDA_ENV     
    params += ' spisonet.py map_refine '
    params += data_file      
    params += ' %s/%s_it%s_3DFSC.mrc' %(dir, basename, var) 
    params += ' --epochs %s --n_subvolume 1000 --acc_batches 2  '   %(epochs) 
    params += ' --output_dir %s' %(dir) 
    params += ' --gpuID %s' %(gpu) 
    if pretrained_model is not None:
        params += ' --pretrained_model %s' %(pretrained_model)
    if mask_file is not None:
        params += ' --mask %s' %(mask_file)

    #params += ' -o %s/relion_external_reconstruct_deep%s.mrc' %(dir, half) 
    #params += ' -g %s -b 5' %gpu
#     params += ' -p wideTarget' 
    print(params)
    os.system(params)

def execute_3dfsc(fn1,fn2,fscn,limit_res=None, mask_file=None): 
    print(CONDA_ENV)    
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
    
if __name__=="__main__":  
    paths = sys.argv 
    star = paths[1]
    print(paths)
    dir=os.path.dirname(star)

    part = star.split('/')[-1].split('_')
    iter_string = part[1]
    basename = part[0]
    iter_number = int(iter_string[2:5])
    print("iter =", iter_number)
    half_str = part[2]
    print('half_str',half_str)
    #if half_str == 'half1':
    if os.getenv('CONDA_ENV'):
        CONDA_ENV=os.getenv('CONDA_ENV')
    else:
        print("Error with conda activation")
    
    if os.getenv('CUDA_VISIBLE_DEVICES'): 
        gpu = os.environ['CUDA_VISIBLE_DEVICES']
        print("gpu = %s" %gpu)  
    else:
        import torch
        gpu_list = list(range(torch.cuda.device_count()))
        gpu=','.join(map(str, gpu_list))
        print("CUDA_VISIBLE_DEVICES not found, using all GPUs in this node: %s" %gpu)  
            
    #We assume iter < 100   
        
    if int(iter_number) <= 9: 
        var = '00%d' %(iter_number)
        beforeVar = '00%d' %(iter_number-1)
    elif int(iter_number) == 10:
        var = '0%d' %(iter_number)
        beforeVar = '00%d' %(iter_number-1)
    else:
        var = '0%d' %(iter_number)
        beforeVar = '0%d' %(iter_number-1)

    with open("%s/%s_it%s_sampling.star" %(dir,basename,beforeVar)) as file:
        for li in file.readlines():
            if "rlnHealpixOrder " in li: 
                healpix = int(li.split()[1]) 
                print("healpix = %s" %healpix)
                break


    mask_file = None
    with open("%s/%s_it%s_optimiser.star" %(dir,basename,beforeVar)) as file:
        for li in file.readlines():
            if "_rlnSolventMaskName " in li: 
                mask_file = li.split()[1]
                print("mask_file = %s" %mask_file)
                break
    
    

    if (healpix < 2):     
        execute_external_relion(star)   
        time.sleep(5)
    else:         
        sampling_index = None
        with open("%s/%s_it%s_data.star" %(dir,basename,beforeVar)) as f:
            for line in f.readlines():
                if "_rlnImagePixelSize" in line:
                    sampling_index = int(line.split()[1].split("#")[1])
                if "opticsGroup1" in line:
                    sampling = float(line.split()[sampling_index-1])
                    print("pixel size = %s" %sampling)  
        
        execute_external_relion(star)   
                
        mrc1 = '%s/%s_it%s_half1_class001_external_reconstruct.mrc' %(dir,basename,var)
        mrc2 = '%s/%s_it%s_half2_class001_external_reconstruct.mrc' %(dir,basename,var)
        mrc_final1 = '%s/%s_it%s_half1_class001_unfil.mrc' %(dir,basename,var)
        mrc_final2 = '%s/%s_it%s_half2_class001_unfil.mrc' %(dir,basename,var)
        model1 = '%s/%s_it%s_half1_class001_external_reconstruct.pt' %(dir,basename,beforeVar)
        model2 = '%s/%s_it%s_half2_class001_external_reconstruct.pt' %(dir,basename,beforeVar)

        check=os.path.isfile(mrc1)        
        check_final = (half_str == "class001") 
        print("check",check)
        print("check_final",check_final)
        if check_final is True:
            mrc1 = mrc_final1
            mrc2 = mrc_final2

        for i in range (1,15):
            try:
                with mrcfile.open(mrc2) as f2:
                    emMap2 = f2.data.astype(np.float32).copy() 
            except:
                print("Waiting for half2")
                time.sleep(30)
            #if (os.path.isfile(mrc1) or os.path.isfile(mrc_final1)) is True:
            #    break
            #else:
            #    time.sleep(30)
                
        # for i in range (1,15):
        #     if (os.path.isfile(mrc2) or os.path.isfile(mrc_final2)) is True:
        #         break
        #     else:
        #         time.sleep(30)
        #time.sleep(30)



        if ((check_final is True) or (check is True)) and (half_str == 'half1' or half_str == 'class001'):                
            fscn='%s/%s_it%s_3DFSC.mrc' %(dir,basename,var)

            with mrcfile.open(mrc1) as f1:
                emMap1 = f1.data.astype(np.float32).copy()  
            with mrcfile.open(mrc2) as f2:
                emMap2 = f2.data.astype(np.float32).copy()   

            with mrcfile.new(mrc1, overwrite=True) as f1:
                f1.set_data(emMap1.astype(np.float32))
                f1.voxel_size = tuple([sampling]*3)
            with mrcfile.new(mrc2, overwrite=True) as f2:
                f2.set_data(emMap2.astype(np.float32))
                f2.voxel_size = tuple([sampling]*3)

            mean1_before =  emMap1.mean()                  
            mean2_before =  emMap2.mean()  
            std1_before =  emMap1.std()                  
            std2_before =  emMap2.std()  
            print("here")
            if check_final is True:
                print("final_iteration, reconstruct 3dfsc to Nyquist %s" %(2*sampling))
                execute_3dfsc(mrc1,mrc2,fscn, limit_res=2*sampling, mask_file=mask_file)
            else:
                execute_3dfsc(mrc1,mrc2,fscn, mask_file=mask_file)

            if check_final is True:
                print("final isonet reconstruction")
                epochs = 10
            else:
                epochs = 5

            if not os.path.isfile(model1):
                print("first isonet reconstruction")
                model1 = None
                model2 = None
                epochs = 10 

            print(f"epochs = {epochs}")
                

            execute_deep(mrc1, dir, basename, var, gpu, epochs = epochs, mask_file = mask_file, pretrained_model = model1)
            execute_deep(mrc2, dir, basename, var, gpu, epochs = epochs, mask_file = mask_file, pretrained_model = model2)
                

            if check is True:
                out_mrc1 = '%s/corrected_%s_it%s_half1_class001_external_reconstruct.mrc' %(dir,basename,var)
                out_mrc2 = '%s/corrected_%s_it%s_half2_class001_external_reconstruct.mrc' %(dir,basename,var)
            else: 
                out_mrc1 = '%s/corrected_%s_it%s_half1_class001_unfil.mrc' %(dir,basename,var)
                out_mrc2 = '%s/corrected_%s_it%s_half2_class001_unfil.mrc' %(dir,basename,var)  



            with mrcfile.open(out_mrc1) as d1:
                    emDeep1 = d1.data.astype(np.float32).copy() 
            with mrcfile.open(out_mrc2) as d2:
                    emDeep2 = d2.data.astype(np.float32).copy()              
            
            finalMap1 = emDeep1*float(std1_before)+mean1_before
            finalMap2 = emDeep2*float(std2_before)+mean2_before
            
            #save mrcfile

            with mrcfile.new(out_mrc1, overwrite=True) as fMap1:
                fMap1.set_data(finalMap1.astype(np.float32))
                fMap1.voxel_size = tuple([sampling]*3)
            with mrcfile.new(out_mrc2 , overwrite=True) as fMap2:
                fMap2.set_data(finalMap2.astype(np.float32))
                fMap2.voxel_size = tuple([sampling]*3)
     
            
