# auto_label_GroundDINO
labler using text prompt 

# requirements
- Nvidia Gpu with CUDA capability (optional, but highly recommended)
 

# installation
#### install grounding dino

*i am using PopOS 22.04 while writing this* <br>
*i have not tested on windows or mac*
> **recommended:** *create and activate a python env*

install nvidia cuda toolkit *(CUDA version 12.2 for reference)*

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
sudo sh cuda_12.2.2_535.104.05_linux.run
```
*your platform might be different [official nvidia page]
(https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local)*

and add to ~/.bashrc file, <br>

**get cuda path with**

`echo $CUDA_HOME`

format goes like this /usr/local/cuda{cudaversion}

*then*

`export CUDA_HOME=path/to/cuda`

```bash 
cd GroundingDINO
pip install -r requirements.txt
pip install -e .
```
#### install SAM

```bash
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
```

#### download weights

```bash

# SAM
wget 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'

#GroundingDINO

wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

```
create a folder called `weights` and then put both the previously downloaded files in
