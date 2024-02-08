# Single Particle spIsoNet Tutorial

Single Particle spIsoNet (spIsoNet) is designed to correct for the preferred orientation problem in cryoEM by self-supervised deep learning, by recovering missing information from well-sampled orientations in Fourier space. Unlike conventional supervised deep learning methods that need explicit input-output pairs for training, spIsoNet autonomously extracts supervisory signals from the original data, ensuring the reliability of the information used for training. 

# 1. Quick run

The default parameter in spIsoNet should be suitable for most cases. You can also train with larger epoch values to obtain better results. 

## 1.0 prepare data set
The tutorial data can be downloaded on google drive: xxx

The tutorial data set contains a two half maps: emd_8731_half_map_1.mrc and emd_8731_half_map_2.mrc and a solvent mask emd_8731_msk_1.mrc. 

After you download the files put those into a new folder.

## 1.1. calculate 3DFSC

The algorithm for 3D FSC is based on
Tan, Y.Z., Baldwin, P.R., Davis, J.H., Williamson, J.R., Potter, C.S., Carragher, B. and Lyumkis, D., 2017. Addressing preferred specimen orientation in single-particle cryo-EM through tilting. Nature methods, 14(8), p.793.

The 3DFSC volume (the default file name is FSC3D.mrc) should be generated in a few minutes. This step does not use GPU accelation. You can use multiple cpu cores for parallelization by specifying "--ncpus".

Thie FSC3D.mrc will be used in the following refine step as "aniso_file"

The input of 3D FSC calculation are two half maps and a solvent mask
``` {.bash language="bash"}
spisonet.py fsc3d emd_8731_half_map_1.mrc emd_8731_half_map_2.mrc emd_8731_msk_1.mrc --ncpus 16 
```

This will generate a 3DFSC volume called "FSC3D.mrc", which describes the Fourier shell correlation in different directions. 

You can also tune the --limit_res parameter to set the resolution limit of the 3D FSC calculation for recovery (default is the overall resolution of the map). 

```
11:07:17, INFO     [spisonet.py:552] Global resolution at FSC=0.143 is 4.191999816894532
11:07:17, INFO     [spisonet.py:555] Limit resolution to 4.191999816894532 for spIsoNet 3D calculation. 
11:07:17, INFO     [spisonet.py:557] calculating fast 3DFSC, this will take few minutes
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [00:02<00:00, 28.95it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [00:03<00:00, 24.20it/s]
11:07:23, INFO     [spisonet.py:562] voxel_size 1.309999942779541
```

## 1.2. Single particle spIsoNet correction of the half maps

This step trains a network for anisotropic correction with "spisonet.py refine". 

This step will create a folder to store the output files of spIsoNet. The corrected map is stored as "correctedXXX.mrc" in that folder. You can also find trained neural network "XX.pt" and figure for loss change "loss.png" in the folder.  

The command to using single particle spIsoNet should be
``` {.bash language="bash"}
spisonet.py refine emd_8731_half_map_1.mrc emd_8731_half_map_2.mrc --aniso_file FSC3D.mrc --mask emd_8731_msk_1.mrc --epochs 30 --alpha 1 --beta 0.5 --output_dir isonet_maps --gpuID 0,1,2,3 --acc_batches 2
```

Here is expected command line output
``` {.bash language="bash"}
11:13:15, INFO     [spisonet.py:466] voxel_size 1.309999942779541
11:13:24, INFO     [map_refine.py:239] Start preparing subvolumes!
11:13:24, INFO     [map_refine.py:242] Done preparing subvolumes!
11:13:24, INFO     [map_refine.py:244] Start training!
11:13:25, INFO     [network.py:202] Port number: 43963
learning rate 0.0003
(8, 9)
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 250/250 [02:13<00:00,  1.87batch/s, Loss=0.598]
Epoch [1/20], Train Loss: 0.6581
 14%|█████████████████████████████████▌                                                                                                                                                                                                       | 36/250 [00:16<01:40,  2.14batch/s, Loss=0.533]
...

```
You can check the command line argument for the map_refine with the following command:
``` {.bash language="bash"}
spisonet.py refine --help
```

## 1.3. Postprocessing

Postprocessing of the corrected halfmaps is not implemented in spIsoNet. Please use your favorite software package (e.g., RELION) for sharpening with the corrected half maps.


# 2. advanced topics

## 2.1 limit resolution
This parameter defined as the resolution limit for spIsoNet recovery. The maps will be first filtered to this resolution for the neural network training. After the network is trained, it will produced a "corrected_xx_filtered.mrc". Then the information beyond this resolution will be added to produce the final results "corrected_xx.mrc".

The higher resolution will introduce unreliable noise that may compromise the results. A lower value may lead to maps with partically recovered missing information. We tested that 3.5A or the resolution at 0.143 could be good starting points to test this value. 

## 2.2 GPU memory consumption and acc_batches
Here is the table of GPU memory consumption. Based on previous spIsoNet experience, larger batch size (> 4) works better. 

acc_batches larger than 1 uses accumulate gradient to reduce memory consumption. Usually acc_batches can be 2 for most cases. If you have GPU with large memory, acc_batches = 1 should process slightly faster. 

batch_size should be divisible by acc_batches.
| Number of GPUs    | batch_size | acc_batches | memory useage per GPU |
| -------- | ------- | ------- | ------- |
| 1        | 4*      | 1*      | ~17.0GB |
| 1        | 4*      | 2       | ~10.7GB |
| 1        | 4*      | 4       | ~7.0GB  |
| 2        | 4*      | 1*      | ~10GB   |
| 4        | 8*      | 1*      | ~10GB   |
| 4        | 8       | 2       | ~5.3GB   |
* means default values

## 2.3 predict directly or continue from a trained model
The single particle spIsoNet will generate a neuronal network model named xx.pt in the output folder, you can start from that model with the parameter "--pretrained_model".

Once you start with the pretrained model, you may also want to change the number of epochs to run. For example, the trained model is from the 10th epochs and you can train for another 10 epochs to make it equivalent to the 20 epochs from scratch.

You can also set --epochs to 0 to only perform prediction without further training


## 2.4 Alpha and Beta weighting

The alpha value defines the weighting between the data consistency loss and the rotational equivarient loss. The default value 1 meaning putting equal weight on the two losses. The larger value means more weight on the rotational equivariant loss.

Empirically, the larger alpha value will results in smoother results. Please see the following images of corrected_half1.mrc with different alpha values.

<p align="center"> <img src="figures/alpha.png" width="800"> </p>

The beta value defines the weighting between missing information recovery and the denoising. The larger value leads to more denoised output maps. The default beta velue is 0.5. We typically do not change this value. 

## 2.5 run spIsoNet with a reliable reference
If you have a low resolution map of your sample that is reliable and with less severe perferred orientation, you can use this as a reference for the spIsoNet refine. This allows you to retain the low resolution information (defined with --ref_resolution parameter) from the reference in the spIsoNet refine process. This may improve the results.

The defaulr --ref_resolution is 10, this resolution should be much lower than the resolution of the reference.

## 2.6 process two half maps independently
The noise2noise-based denoising is by default used in the spIsoNet refine process. It will produce cleaner and better maps. However, this denoising network will see both half1 and half2, which will break the independency of the two half maps.

You can specify the "--independent" as True to run the two half map independently, which will only perform missing information recovery without denoising. This step will train two networks each for one half map. The results generated from "--independent" reconstruction can be used for the "gold-standard" FSC calculation.  

# 3. spIsoNet enpowered Relion refine
The particle alignment will nevertheless be influenced by distorted map with perferred orientation. This step is very useful in terms of generate a better alignment and subsqeuence a better cryoEM map.

This deep-learning approach spIsoNet can be used as a regularizer in the RELION refinement process. In each iteration of RELION refinement, spIsoNet can be used to perform the 3D reconstruction to generate corrected map and use that map for orientation search in RELION refine process.

To achieve this, the following steps should be performed. 

## 3.1 Make sure spIsoNet is properly installed in a conda environment.

Install the required dependencies
```
conda create -n spisonet python=3.10
conda activate spisonet 
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirments.txt
```

and set up proper environment for spIsoNet. It is required to set the RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE environment variable to point to relion_wrapper.py.
```
export PATH=/home/cii/software/spIsoNet/bin:$PATH
export PYTHONPATH=/home/cii/software:$PYTHONPATH
export RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE="python /home/cii/software/spIsoNet/bin/relion_wrapper.py"
export CONDA_ENV="spisonet"
```

## 3.2 Execute relion_wrapper.py script in RELION's relion_refine.
To execute the script relion_wrapper.py in relion_refine, it is necessary to add the argument "--external_reconstruct" in the command line, or by adding "--external_reconstruct" in the Additional Arguments section under the Running tab in RELION GUI.

Here is an example command:
``` {.bash language="bash"}
mpirun -np 5 `which relion_refine_mpi` --o Refine3D/job001/run --auto_refine --split_random_halves --i particles.star --ref reference.mrc --firstiter_cc --ini_high 10 --dont_combine_weights_via_disc --preread_images  --pool 30 --pad 2  --ctf --particle_diameter 170 --flatten_solvent --zero_mask --solvent_mask mask.mrc --solvent_correct_fsc  --oversampling 1 --healpix_order 2 --auto_local_healpix_order 5 --offset_range 5 --offset_step 2 --sym C3 --low_resol_join_halves 40 --norm --scale  --j 4 --gpu "" --external_reconstruct --combine --pipeline_control Refine3D/job001/
```

Here is the place for "--external_reconstruct":
<p align="center"> <img src="figures/external.png" width="800"> </p>

To use the spIsoNet, the relion refine command shoule contains:
1. "--external_reconstruct"
2. "--solvent_mask"
3. "--solvent_correct_fsc" 

If you have a reference that have a lower resolution but without perferred orientation. You can specify the "--combine" parameter, this command line argument is not recognized by relion bur can be recognized by spIsoNet. The limitation to retain the low resolution information from the reference is set by "--ini_high" paramter. With this parameter, the perfermance of the spIsoNet recovery and denoising might be improved.

In some severe cases, the "--combine" is necessaey. When the map generated by relion is not correct, it is important to keep the low resolution information from a correct reference throughout the image alignment. 


## 3.3 Environment variables for the relion wrapper.

To tune the parameter of spIsoNet enpowered relion refinement, you can set up more linux environemnt variables as follows:

1. set which GPU(s) you can use by sepecify the CUDA_VISIBLE_DEVICES. By default, spIsoNet will use up all the avaliable GPUs 
```
export CUDA_VISIBLE_DEVICES="0"
```

2. ISONET_BETA value will define the denoising level of the network, setting this to zero will prevent the noise2noise based denoising

```
export ISONET_BETA=0.5
```

3. ISONET_ALPHA value define the balance between the data consistency loss and rotation equivariance loss. Default value 1 should work for most cases. Setting this to zero is not recommand and will ignore rotation equivariance loss. 
```
export ISONET_ALPHA=1
```

4. START_HEALPIX defines at which angular sampling step the spIsoNet training will be performed.
```
ISONET_START_HEALPIX=3
```

5. ISONET_RETRAIN_EACH_ITER defines whether the network should be retrained from scratch or from previous iteration of relion refine. This is typically set to True
```
ISONET_RETRAIN_EACH_ITER=True
```

6. ISONET_EPOCHS defines how many epochs to train the neural network.
```
ISONET_EPOCHS=10
```

7. ISONET_ACC_BATCHES defines how many small batches for each batch. If you have GPUs with large memory you can set it to 1.
```
ISONET_ACC_BATCHES=2
```

8. ISONET_COMBINE define whether low resolution information from a correct reference is kept in the alignment. This parameter can be overwrite by the --combine parameter in relion command or GUI
```
ISONET_COMBINE=False
```

9. ISONET_COMBINE define whether low resolution information from a correct reference is kept in the alignment. This parameter can be overwrite by the --combine parameter in relion command or GUI
```
ISONET_COMBINE=False
```


10. ISONET_WHITENING define whether whighting is performed before running spIsoNet. Typical we set this as True.
ISONET_WHITENING_LOW defines the starting resolution for whitening.
```
ISONET_WHITENING=True
ISONET_WHITENING_LOW=10

```
11. ISONET_FSC_WEIGHTING define whether FSC weighting is performed before running spIsoNet. Typical we set this as True.
```
ISONET_FSC_WEIGHTING=True
```
