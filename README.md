
# spIsoNet version 1.0

Update on Mar.27 2024

Single Particle spIsoNet (spIsoNet) is designed to correct for the preferred orientation problem in cryoEM by self-supervised deep learning, by recovering missing information from well-sampled orientations in Fourier space. 

Unlike conventional supervised deep learning methods that need explicit input-output pairs for training, spIsoNet autonomously extracts supervisory signals from the original data, ensuring the reliability of the information used for training.

spIsoNet is designed for single particle analysis and subtomogram averaging. For the correcting missing wedge in cryoET, please refer to IsoNet.

Please find tutorial/spIsoNet_v1.0_Tutorial.md for detailed document.

## Google group
We maintain an spIsoNet Google group for discussions or news.

To subscribe or visit the group via the web interface please visit https://groups.google.com/u/1/g/spisonet. 

To post to the forum you can either use the web interface or email to spisonet@googlegroups.com

# Installation

We suggest using anaconda environment to manage the spIsoNet package.

Example commands to install spIsoNet

*Option 1:*
```
conda create -n spisonet python=3.10
conda activate spisonet
pip install torch --index-url https://download.pytorch.org/whl/cu118
cd <path to spIsoNet>
pip install .
```
and then set the following environment variable for Misalignment Correction
```
export RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE="python <path to spIsoNet>/spIsoNet/bin/relion_wrapper.py"
export CONDA_ENV="spisonet"
```

*Option 2:*
```
conda env create -f setup.yml
conda activate spisonet
```

and then set the following environment variable for Misalignment Correction
```
export RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE="python <path to spIsoNet>/spIsoNet/bin/relion_wrapper.py"
export CONDA_ENV="spisonet"
```

*Option 3:*
```
conda create -n spisonet python=3.10
conda activate spisonet
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

and then set the following environment variable for Misalignment Correction
```
export RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE="python <path to spIsoNet>/spIsoNet/bin/relion_wrapper.py"
export CONDA_ENV="spisonet"
export PATH=<path to spIsoNet>/spIsoNet/bin:$PATH
export PYTHONPATH=<path to spIsoNet>:$PYTHONPATH
```


The environment we verified are:
1. cuda11.8 cudnn8.5 pytorch2.0.1, pytorch installed with pip.
2. cuda11.3 cudnn8.2 pytorch1.13.1, pytorch installed with conda.