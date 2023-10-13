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

# import fcntl
# import errno

#for isonet 
'''
need GPU information 
'''
def skew_towards_edges(x, exponent=0.5):
    # Ensure x is within the range -1 to 1
    x = (float(x)-90)/90.

    x = np.clip(x, -1, 1)
    # Apply the skewing function
    if x < 0:
        result = -abs(x) ** exponent
    else:
        result = x ** exponent
    return result*90+90

def rebalance(star):
    s = ""
    with open(star) as file:
        for line_number,li in enumerate(file.readlines()):
            if "_rlnAngleTilt " in li:
                tlt_index = int(li.split()[1].split("#")[1])
            if ".mrc" in li:
                print(li)
                split_li = li.split()
                tlt = skew_towards_edges(split_li[tlt_index-1], 0.8)
                split_li[tlt_index-1] = str(tlt)
                s+=" ".join(split_li)
            else:
                s+=li.strip()
            s += "\n"
    with open(star, 'w') as file:
        file.write(s)

def execute_external_relion(star):

    params = ' relion_external_reconstruct'
    params += ' %s' %star
    os.system(params)     
    
def execute_deep(data_file,fsc3d, dir, gpu, epochs = 1, mask_file = None, pretrained_model=None, alpha = None, acc_batches= None, batch_size = None): 
    #data_file =  ' %s/%s_it%s_half%s_class001_external_reconstruct.mrc' %(dir, basename, var, half)   
    print(f"processing {data_file}")  
    params = ' eval "$(conda shell.bash hook)" && conda activate %s && ' %CONDA_ENV     
    params += ' spisonet.py map_refine '
    params += data_file      
    params += ' %s' %(fsc3d) 
    params += ' --epochs %s --n_subvolume 1000'   %(epochs)
    if acc_batches is not None:
        params += ' --acc_batches %s'   %(acc_batches) 
    if batch_size is not None:
        params += ' --batch_size %s'   %(batch_size) 
    if alpha is not None:
        params += ' --alpha %s'   %(alpha) 
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

    if os.getenv('ISONET_START_HEALPIX'): 
        limit_healpix = os.environ['ISONET_START_HEALPIX']
        limit_healpix = int(limit_healpix)
        print("start_healpix = %s" %limit_healpix)  
    else:
        limit_healpix = 3


    alpha = os.getenv("ISONET_ALPHA")
    if alpha is None:
        alpha = 1
    print(f"alpha: {alpha}")
    acc_batches = os.getenv("ISONET_ACC_BATCHES")
    if acc_batches is None:
        acc_batches = 1
    epochs = os.getenv('ISONET_EPOCHS')
    if epochs is None:
        epochs = 5

    model1 = os.getenv('ISONET_MODEL_FILE1')
    model2 = os.getenv('ISONET_MODEL_FILE2')
    fsc3d = os.getenv('ISONET_3DFSC')

    start_epochs = os.getenv('ISONET_START_EPOCHS')
    if start_epochs is None:
        start_epochs = epochs
    batch_size = os.getenv("ISONET_BATCHSIZE")

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
    
    use_unfil = True
    if use_unfil==True:
        ext = "unfil"
    else:
        ext = "external_reconstruct"
    
    if model1 is None:
        model1 = '%s/%s_it%s_half1_class001_%s.pt' %(dir,basename,beforeVar,ext)
        model2 = '%s/%s_it%s_half2_class001_%s.pt' %(dir,basename,beforeVar,ext) 
    #model1 = '%s/%s_it%s_half1_class001_unfil.pt' %(dir,basename,beforeVar)
    #model2 = '%s/%s_it%s_half2_class001_unfil.pt' %(dir,basename,beforeVar) 
    check_final = (half_str == "class001") 

 
    debug_mode =True
    if (healpix < limit_healpix):   
        shutil.copy(star, star+".backup")  
        rebalance(star)
        execute_external_relion(star)  
        time.sleep(5) 
        #rebalance("'%s/%s_it%s_data.star' %(dir,basename,beforeVar)")   


    elif (check_final is True):
        execute_external_relion(star)
        mrc_final1 = '%s/%s_half1_class001_unfil.mrc' %(dir,basename)
        mrc_final2 = '%s/%s_half2_class001_unfil.mrc' %(dir,basename)
        final_fsc = '%s/%s_3DFSC.mrc' %(dir,basename)
        print("-----------")
        print("Important information for the final iteration!!!")
        print("The final half_unfil maps are generated with relion")
        print("You may want to further use IsoNet to correct the final maps")
        print("spIsoNet commands to correct final maps are:")
        print(f"spisonet.py fsc3d {mrc_final1} {mrc_final2} -o {final_fsc}")
        print(f"spisonet.py map_refine {mrc_final1} {final_fsc} --output_dir {dir} --pretrained_model {model1} --mask {mask_file}")
        print(f"spisonet.py map_refine {mrc_final2} {final_fsc} --output_dir {dir} --pretrained_model {model2} --mask {mask_file}")
    else:         
        execute_external_relion(star)
        mrc1_overwrite = '%s/%s_it%s_half1_class001_external_reconstruct.mrc' %(dir,basename,var)
        mrc2_overwrite = '%s/%s_it%s_half2_class001_external_reconstruct.mrc' %(dir,basename,var)
        mrc1 = '%s/%s_it%s_half1_class001_%s.mrc' %(dir,basename,var,ext)
        mrc2 = '%s/%s_it%s_half2_class001_%s.mrc' %(dir,basename,var,ext)

        if (half_str == 'half1'):
            for i in range (1,15):
                try:
                    with mrcfile.open(mrc2) as f2:
                        pass#f2.data.astype(np.float32).copy() 
                    with mrcfile.open(mrc2_overwrite) as f2:
                        pass#f2.data.astype(np.float32).copy() 
                except:
                    print("Waiting for half2")
                    time.sleep(30)

            sampling_index = None
            with open("%s/%s_it%s_data.star" %(dir,basename,beforeVar)) as f:
                for line in f.readlines():
                    if "_rlnImagePixelSize" in line:
                        sampling_index = int(line.split()[1].split("#")[1])
                    if "opticsGroup1" in line:
                        sampling = float(line.split()[sampling_index-1])
                        print("pixel size = %s" %sampling) 

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
            print('Resolution in previous iteration', resolution)
            print('limit resolution to FSC=0.143', limit_resolution)
            if limit_resolution < 20:
                if (fsc3d is None) or (len(fsc3d)<2):
                    fscn='%s/%s_it%s_3DFSC.mrc' %(dir,basename,var)    
                    execute_3dfsc(mrc1,mrc2,fscn, limit_res=limit_resolution, mask_file=mask_file)    
                else:
                    fscn = fsc3d
                print(f"using FSC3D file {fscn}")

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


                if not os.path.isfile(model1):
                    print(f"first isonet reconstruction, because previous iteration healpix order become {limit_healpix}")
                    model1 = None
                    model2 = None

                    
                out_mrc1 = '%s/corrected_%s_it%s_half1_class001_%s.mrc' %(dir,basename,var,ext)
                out_mrc2 = '%s/corrected_%s_it%s_half2_class001_%s.mrc' %(dir,basename,var,ext)

                if debug_mode:
                    execute_deep(mrc1,fscn, dir, gpu, epochs = 0, mask_file = mask_file, pretrained_model = model1, batch_size = batch_size, acc_batches=acc_batches, alpha=alpha)
                    execute_deep(mrc2,fscn, dir,  gpu, epochs = 0, mask_file = mask_file, pretrained_model = model2, batch_size = batch_size, acc_batches=acc_batches, alpha=alpha)

                    shutil.move(out_mrc1, '%s/prepredicted_%s_it%s_half1_class001_%s.mrc' %(dir,basename,var,ext))
                    shutil.move(out_mrc2, '%s/prepredicted_%s_it%s_half2_class001_%s.mrc' %(dir,basename,var,ext))

                if model1 is not None:
                    execute_deep(mrc2,fscn, dir,  gpu, epochs = epochs, mask_file = mask_file, pretrained_model = model2, batch_size = batch_size, acc_batches=acc_batches, alpha=alpha)
                    execute_deep(mrc1,fscn, dir,  gpu, epochs = epochs, mask_file = mask_file, pretrained_model = model1, batch_size = batch_size, acc_batches=acc_batches, alpha=alpha)
                else:
                    execute_deep(mrc2,fscn, dir,  gpu, epochs = start_epochs, mask_file = mask_file, pretrained_model = model2, batch_size = batch_size, acc_batches=acc_batches, alpha=alpha)
                    execute_deep(mrc1,fscn, dir,  gpu, epochs = start_epochs, mask_file = mask_file, pretrained_model = model1, batch_size = batch_size, acc_batches=acc_batches, alpha=alpha)                    
                
                if debug_mode:
                    shutil.copy(mrc1, '%s/precorrect_%s_it%s_half1_class001_%s.mrc' %(dir,basename,var,ext))
                    shutil.copy(mrc2, '%s/precorrect_%s_it%s_half2_class001_%s.mrc' %(dir,basename,var,ext))   

                with mrcfile.open(out_mrc1) as d1:
                    emDeep1 = d1.data.astype(np.float32).copy() 
                with mrcfile.open(out_mrc2) as d2:
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
                if not debug_mode:
                    os.remove(out_mrc1)
                    os.remove(out_mrc2)  

                with mrcfile.open(fscn, 'r') as mrc:
                    fsc = mrc.data

                from spIsoNet.util.FSC import recommended_resolution
                res = recommended_resolution(fsc, sampling, 0.143)
                print(f"3DFSC resolution {res}")
                from subprocess import check_output
                s = f"relion_image_handler --i {mrc1_overwrite} --o tmp.mrc --lowpass {res}; mv tmp.mrc {mrc1_overwrite}"
                check_output(s, shell=True)
                s = f"relion_image_handler --i {mrc2_overwrite} --o tmp.mrc --lowpass {res}; mv tmp.mrc {mrc1_overwrite}"
                check_output(s, shell=True)

                print("finished spisonet reconstruction")
            else:
                print("skip this iteration of spIsoNet")
     
            
