# Requirements for peft training (LORA, 8-bit quantization, etc.)

`conda create -n peft ipykernel ipywidgets python=3.10 --yes`  
Always add ipykernel and ipywidgets to your environments.

`conda activate peft`

`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`  
This installs multiple libcudarts. Which makes bitsandbytes unhappy, but this is also their recommended way of installing CUDA. I couldn't find a solution.
`UserWarning: Found duplicate ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] files: {PosixPath('/home/jovyan/conda_envs/peft/lib/libcudart.so'), PosixPath('/home/jovyan/conda_envs/peft/lib/libcudart.so.11.0')}`

`conda install transformers datasets scikit-learn -c huggingface --yes`

`pip install peft wandb bitsandbytes sentencepiece`