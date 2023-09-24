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
# export TF_FORCE_GPU_ALLOW_GROWTH='true'

# In order to run deepEMhancer it is necessary to activate 
# the conda environment used in deepEMhancer installation. 


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
halfmap1 and 2
optionally we need a mask
need GPU information 
'''

def execute_external_relion(star):

    params = ' relion_external_reconstruct'
    params += ' %s' %star
    os.system(params)     
    
def execute_deep(dir, basename, var, half, epochs = 1, mask_file = None, pretrained_model=None): 
    print(CONDA_ENV) 
    data_file =  ' %s/%s_it%s_half%s_class001_external_reconstruct.mrc' %(dir, basename, var, half)   
    print(f"processing {data_file}")  
    params = ' eval "$(conda shell.bash hook)" && conda activate %s && ' %CONDA_ENV     
    params += ' spisonet.py map_refine '
    params += data_file      
    params += ' %s/%s_it%s_3DFSC.mrc' %(dir, basename, var) 
    params += ' --epochs %s --n_subvolume 1000'   %(epochs) 
    params += ' --output_dir %s' %(dir) 
    if pretrained_model is not None:
        params += ' --pretrained_model %s' %(pretrained_model)
    if pretrained_model is not None:
        params += ' --mask %s' %(mask_file)

    #params += ' -o %s/relion_external_reconstruct_deep%s.mrc' %(dir, half) 
    #params += ' -g %s -b 5' %gpu
#     params += ' -p wideTarget' 
    print(params)
    os.system(params)

def execute_3dfsc(fn1,fn2,fscn): 
    print(CONDA_ENV)    
    params = ' eval "$(conda shell.bash hook)" && conda activate %s && ' %CONDA_ENV     
    params += ' spisonet.py fsc3d '
    params += ' %s' %(fn1)  
    params += ' %s' %(fn2) 
    params += ' -o %s' %(fscn)
    
    params += ' --limit_res 2.488'
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
        gpu = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0] 
        print("gpu = %s" %gpu)  
    else:
        gpu = 0
            
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

    mask_file = None
    with open("%s/%s_it%s_optimiser.star" %(dir,basename,beforeVar)) as file:
        for li in file.readlines():
            if "_rlnSolventMaskName " in li: 
                mask_file = li.split()[1]
                print("mask_file = %s" %mask_file)
    
    
    # if iter_number < 1:
    #if (healpix < 4):
    if (healpix < 4):     
        execute_external_relion(star)   
        time.sleep(5)
        
    else:         
            ###sampling of the images            
        with open("%s/%s_it%s_data.star" %(dir,basename,beforeVar)) as f:
            for line in f.readlines():
                if "opticsGroup1" in line:
                    sampling = float(line.split()[8])
                    #print("sampling = %s" %sampling)
        
        
        try:        
            file1=os.path.isfile('%s/relion_external_reconstruct_deep1.mrc' %(dir))
            if file1 is True:
                os.remove('%s/relion_external_reconstruct_deep1.mrc' %(dir))
                file1 = False
        except:
            pass        
        
        try:
            file2=os.path.isfile('%s/relion_external_reconstruct_deep2.mrc' %(dir))
            if file2 is True:
                os.remove('%s/relion_external_reconstruct_deep2.mrc' %(dir)) 
                file2 = False
        except:
            pass        
        
        execute_external_relion(star)   
                
        for i in range (1,15):
            mrc1 = os.path.isfile('%s/%s_it%s_half1_class001_external_reconstruct.mrc' %(dir,basename,var))  
            if mrc1 is True:
                break
            else:
                time.sleep(30)
                
        for i in range (1,15):
            mrc2 = os.path.isfile('%s/%s_it%s_half2_class001_external_reconstruct.mrc' %(dir,basename,var))  
            if mrc2 is True:
                break
            else:
                time.sleep(30)

        check=os.path.isfile('%s/%s_it%s_half1_class001_external_reconstruct.mrc' %(dir,basename,var))        
        check_final = os.path.isfile('%s/%s_it%s_half1_class001_unfil.mrc' %(dir,basename,var))  
        print("check",check)
        print("check_final",check_final)
        if ((check_final is True) or (check is True)) and half_str == 'half1':                
        
            #open mrcfile
            fn1='%s/%s_it%s_half1_class001_external_reconstruct.mrc' %(dir,basename,var)
            fn2='%s/%s_it%s_half2_class001_external_reconstruct.mrc' %(dir,basename,var)
            fscn='%s/%s_it%s_3DFSC.mrc' %(dir,basename,var)
            with mrcfile.open(fn1) as f1:
                emMap1 = f1.data.astype(np.float32).copy()  
            with mrcfile.open(fn2) as f2:
                emMap2 = f2.data.astype(np.float32).copy()   
            
            mean1_before =  emMap1.mean()                  
            mean2_before =  emMap2.mean()  
            std1_before =  emMap1.std()                  
            std2_before =  emMap2.std()  
            print("here")
            execute_3dfsc(fn1,fn2,fscn)
            model_file1 = '%s/%s_it%s_half1_class001_external_reconstruct.pt' %(dir,basename,beforeVar)
            if os.path.isfile(model_file1):
                execute_deep(dir, basename, var, '1', mask_file, model_file1)
            else:
                execute_deep(dir, basename, var, '1', mask_file)     

            model_file2 = '%s/%s_it%s_half1_class001_external_reconstruct.pt' %(dir,basename,beforeVar)
            if os.path.isfile(model_file2):
                execute_deep(dir, basename, var, '2', mask_file, model_file2)
            else:
                execute_deep(dir, basename, var, '2', mask_file)    

            

            with mrcfile.open('%s/corrected_%s_it%s_half1_class001_external_reconstruct.mrc' %(dir,basename,var)) as d1:
                emDeep1 = d1.data.astype(np.float32).copy() 
            with mrcfile.open('%s/corrected_%s_it%s_half1_class001_external_reconstruct.mrc' %(dir,basename,var)) as d2:
                emDeep2 = d2.data.astype(np.float32).copy()

            #max1_after = emDeep1.max()
            #max2_after = emDeep2.max()
        
            #factor1 = float(max1_after)/float(max1_before) 
            #factor2 = float(max2_after)/float(max2_before)
            
            finalMap1 = emDeep1*float(std1_before)+mean1_before
            finalMap2 = emDeep2*float(std2_before)+mean2_before
            
            #save mrcfile
            with mrcfile.new('%s/%s_it%s_half1_class001_external_reconstruct.mrc' %(dir,basename,var) , overwrite=True) as fMap1:
                fMap1.set_data(finalMap1.astype(np.float32))
                fMap1.voxel_size = tuple([sampling]*3)
            with mrcfile.new('%s/%s_it%s_half2_class001_external_reconstruct.mrc' %(dir,basename,var) , overwrite=True) as fMap2:
                fMap2.set_data(finalMap2.astype(np.float32))
                fMap2.voxel_size = tuple([sampling]*3)