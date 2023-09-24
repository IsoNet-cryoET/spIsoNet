[![DOI](https://zenodo.org/badge/222662248.svg)](https://zenodo.org/badge/latestdoi/222662248)
##
# Isotropic Reconstruction of Electron Tomograms with Deep Learning
# spIsoNet version 1.0

Update on June27 2023

## Google group
We maintain an spIsoNet Google group for discussions or news.

To subscribe or visit the group via the web interface please visit https://groups.google.com/u/1/g/isonet. 

If you do not have and are not willing to create a Google login, you can also request membership by sending an email to yuntao@g.ucla.edu

To post to the forum you can either use the web interface or email to isonet@googlegroups.com

## Major changes from provious version of spIsoNet
1. spIsoNet 1.0 incoorporated single particle spIsoNet (spspIsoNet) package, which enables correct artifacts caused by preffered orientaion.

To run spIsoNet for single particle analysis. Please run command like 
```
spisonet.py map_refine half1.mrc half2.mrc mask.mrc
```

2. The network become larger and can not be modified 


## Installation

python version at least 3.5 is required. If you download the package as a zip file from github, please rename the folder spIsoNet-master to spIsoNet.

1.  spIsoNet relies on Tensorflow with version at least 2.0

Please find your cuda version, cuDNN version and corresponding tensorflow version here: https://www.tensorflow.org/install/source#gpu. 

For example, if you are using cude 10.1, you should install tensorflow 2.3:
```
pip install tensorflow-gpu==2.3.0
```

2.  Install other dependencies

```
pip install -r requirements.txt
```
3.  Add environment variables: 

For example add following lines in your ~/.bashrc
```
export PATH=PATH_TO_ISONET_FOLDER/bin:$PATH 

export PYTHONPATH=PATH_TO_PARENT_FOLDER_OF_ISONET_FOLDER:$PYTHONPATH 
```
4. Open a new terminal, enter your working directory and run 
```
spisonet.py check
```

Tutorial data set and tutorial videos are on google drive https://drive.google.com/drive/folders/1DXjIsz6-EiQm7mBMuMHHwdZErZ_bXAgp

# FAQ:
## 1. spIsoNet refine raise OOM error.

This is caused by the insufficient GPU memory.
The soluitions are:
1. Specify a smaller batch\_size or use more(powerful) GPUs. The default batch\_size is 4 if you use one GPU, otherwise the default batch\_size is 2 times the number of GPU. Please note the batch_size should be divisiable by number of GPUs.
For example, if you have one GPU and get OOM error, please reduce the batch\_size to 1 or 2; If you use 4 GPUs and get OOM error, please reduce the batch\_size to 4.

2. Refine with a smaller cube\_size (not recommanded).

## 2.  spIsoNet extract ValueError: a must be greater than 0 unless no samples are taken
This could be due to the tomogram thickness is smaller than the size of subtomograms to be extracted. Please make your tomogram thicker in this case.

## 3. Can not see significent improvement after processing with spIsoNet
spIsoNet is kind of conservative in adding information into missing wedge region. If it can not find reasonable prediction, spIsoNet may simply returns the origional tomograms back to you. 
However, there are some ways to increase your success rate.
1. spIsoNet performs better in high contrast tomograms. That means it will be helpful to tweak the parameters (especially snrfalloff) in CTF deconvolution step to make increase the weight of low resolution information. Or trying with the data acquired with phaseplate first. As far as we know, phaseplate data will always give you good result.

2. Missing wedge caused the nonlocal distributted information. You may observed the long shadows of gold beads in the tomograms, and those long rays can not be fully corrected with sub-tomogram based missing correction in spIsoNet, because the receptive field of the network is limitted to your subtomogram. This nonlocal information makes it particular difficult to recover the horizontal oriented membrane. There are several ways to improve. **First**, training with subtomograms with larger  cube size, the default cube size is 64, you may want to increase the size to 80, 96, 112 or 128, however this may lead to the OOM error Please refer to FAQ #1 when you have this problem. **Second**, bin your tomograms more. Some times we even bin our celluar tomograms to 20A/pix for spIsoNet processing, this will of course increase your network receptive field, given the same size of subtomogram. 

3. spIsoNet is currently designed to correct missing wedge for tomograms with -60 to 60 degress tilt range. The other tilt scheme or when the tomograms have large x axis tilt. The results might not be optimal. 
## 4. Can not create a good mask during mask generation step
The mask is only important if the sample is sparsely located in the tomograms. And the mask do not need to be perfect to obtain good result, in other words, including many empty/unwanted subtomos during the refinement can be toralated. 

To obtain a good mask, the tomograms should have sufficient contrast, which can be achieved by CTF deconvolution. User defined mask can also be supplied by changing the mask_name field in the star file. Alternately, you can also use subtomograms extracted with other methods and skip the entire mask creation and subtomograms extraction steps.

If you want to exclude carbon area of the tomograms, you can try the new mask boundary feature in version 0.2. It allows you to draw a polygon in 3dmod so that the area outside the polygon will be excluded.

4. Error loding GUI:
The following error might occured when running spisonet.py gui: 
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Solution: please install dependencies
yum install xcb-util*
yum install libxkbcommon-x11
