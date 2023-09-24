# Single Particle spIsoNet Tutorial

Single Particle spIsoNet (spspIsoNet) is designed to correct the preffered orientation effect by self-supervised learning. It iteratively reconstruct missing information in in the missing regions in fourier space. The software requires half maps as input.



# 1. Installation

We suggest using anaconda environment to manage the spIsoNet package.

1. install cudatoolkit and cudnn on your computer.
2. Install pytorch from https://pytorch.org/ 
3. install dependencies using pip install
   the dependencies include tqdm, matplotlib, scipy, numpy, scikit-image, mrcfile, fire
4. For example add following lines in your ~/.bashrc

    export PATH=PATH_TO_ISONET_FOLDER/bin:$PATH 

    export PYTHONPATH=PATH_TO_PARENT_FOLDER_OF_ISONET_FOLDER:$PYTHONPATH 
    or you can run source source-env.sh in your terminal, which will export required variables into your environment.
5. Now spIsoNet is avaliable to use.

The environment we verified are:
1. cuda11.8 cudnn8.5 pytorch2.0.1, pytorch installed with pip.
2. cuda11.3 cudnn8.2 pytorch1.13.1, pytorch installed with conda.


# 2. Quick run

The default parameter in spIsoNet should be suitable for most cases, you can also train with a larger epochs to get better result. 

## 2.0 prepare data set
The tutorial data can be downloaded on google drive: xxx
The tutorial data set contains a two half maps: emd_8731_half_map_1.mrc and emd_8731_half_map_2.mrc and a solvent mask emd_8731_msk_1.mrc. After you download the files put those into a new folder.



## 2.1. make 3D FSC

The algorithum for 3D FSC is based on
Tan, Y.Z., Baldwin, P.R., Davis, J.H., Williamson, J.R., Potter, C.S., Carragher, B. and Lyumkis, D., 2017. Addressing preferred specimen orientation in single-particle cryo-EM through tilting. Nature methods, 14(8), p.793.

This 3D FSC reimplementation in single particle spIsoNet should be faster than the origional version. The 3DFSC volume should be renerated in few minutes. This step does not use GPU accelations. You can use multiple cpu cores for parallelation by specifying "--ncpus".

The input of 3D FSC calculation are two half maps and a solvent mask
``` {.bash language="bash"}
spisonet.py fsc3d emd_8731_half_map_1.mrc emd_8731_half_map_2.mrc emd_8731_msk_1.mrc --ncpus 16 
```

This will generate a 3DFSC volume called "FSC3D.mrc", which describes the Fouriear shell correlation in different directions. 

```
11:07:17, INFO     [spisonet.py:552] Global resolution at FSC=0.143 is 4.191999816894532
11:07:17, INFO     [spisonet.py:555] Limit resolution to 4.191999816894532 for spIsoNet 3D calculation. You can also tune this paramerter with --limit_res .
11:07:17, INFO     [spisonet.py:557] calculating fast 3DFSC, this will take few minutes
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [00:02<00:00, 28.95it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [00:03<00:00, 24.20it/s]
11:07:23, INFO     [spisonet.py:562] voxel_size 1.309999942779541
```

## 2.2. Single particle spIsoNet correction of the half map1

This step used train a network for anositropic correction for first half map with "spisonet.py map_refine". The input of this command is the first halfmap and the solvent mask.

This step will create a folder to store the output files of spIsoNet. The corrected map is stored as "correctedXXX.mrc" in that folder. You can also find trained neural network "XX.pt" and figure for loss change "loss.png" in the folder.  

The command to using single particle spIsoNet should be
``` {.bash language="bash"}
spisonet.py map_refine emd_8731_half_map_1.mrc FSC3D.mrc --mask emd_8731_msk_1.mrc --epochs 20 --alpha 1 --output_dir isonet_maps --gpuID 0,1,2,3
```

Here is expected command line output
``` {.bash language="bash"}
11:13:15, INFO     [utils.py:15] The isonet_maps folder already exists, outputs will write into this folder
11:13:15, INFO     [spisonet.py:466] voxel_size 1.309999942779541
11:13:15, WARNING  [utils.py:8] The isonet_maps/data folder already exists. The old isonet_maps/data folder will be moved to isonet_maps/data~
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
spisonet.py map_refine --help
```

## 2.3. Single particle spIsoNet correction of the half map2

This step is performed similar with the previous step, but instead of providing the half1, you would provide half2 as command line argument.

The example command of this step 
``` {.bash language="bash"}
spisonet.py map_refine emd_8731_half_map_2.mrc FSC3D.mrc --mask emd_8731_msk_1.mrc --epochs 20 --alpha 1 --output_dir isonet_maps --gpuID 0,1,2,3
```

## 2.4. Postprocessing

Postprocessing of the corrected halfmaps is not implimented in single particle spIsoNet. Please use your favourate software package for sharpening with the corrected half maps.


# 3. advanced topics

## 3.1 Put all steps together.

You can prepare a sh file containing three commands and run them in a single step: 
such as 
```
spisonet.py fsc3d emd_8731_half_map_1.mrc emd_8731_half_map_2.mrc emd_8731_msk_1.mrc --ncpus 16
spisonet.py map_refine emd_8731_half_map_1.mrc FSC3D.mrc --mask emd_8731_msk_1.mrc --gpuID 0,1,2,3
spisonet.py map_refine emd_8731_half_map_1.mrc FSC3D.mrc --mask emd_8731_msk_1.mrc --gpuID 0,1,2,3
```

## 3.2 GPU memory consumption
Here is the table of GPU memory consumption. Based on previous spIsoNet experience, larger batch size (> 4) works better. And acc_batches larger than 1 uses accumulate gradient to reduce memory consumption and batch_size should be divisible by acc_batches.
| Number of GPUs    | batch_size | acc_batches | memory useage per GPU |
| -------- | ------- | ------- | ------- |
| 1        | 4*      | 1*      | ~17.0GB |
| 1        | 4*      | 2       | ~10.7GB |
| 1        | 4*      | 4       | ~7.0GB  |
| 2        | 4*      | 1*      | ~10GB   |
| 4        | 8*      | 1*      | ~10GB   |
* means default values

## 3.3 continue from previous trained model
The single particle spIsoNet will generate a neuronal network model named xx.pt in the output folder, you can start from that model with the parameter "--pretrained_model".

Once you start with the pretained model, you may also want to change the number of epoches to run. For example, the trained model is from the 10th epochs and you can train for another 10 epoch to make it equivalent to the 20 iteractions from scratch.


## 3.4 Alpha weighting

The most importent parameter that affect your result in single particle spIsoNet is alpha. This alpha defines the weighting between the data consistency loss and the rotational equivarient loss. The default value one meaning putting equal weight on the two losses. The larger value means more weight on the rotational equivarient loss.

Empirically, the larger alpha value will results in smoother results. Please see the following images of corrected_half1.mrc with different alpha values.

<p align="center"> <img src="figures/alpha.png" width="800"> </p>