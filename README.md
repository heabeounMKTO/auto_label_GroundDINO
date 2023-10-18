# auto_label_GroundDINO
labler using text prompt 


# installation
#### install grounding dino

*create and activate a python env*

install nvidia cuda toolkit and add to ~/.bashrc file, <br>

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