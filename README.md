## Machine translation using Huggingface t5 model with fine-tuning

### Dataset
The dataset contains translation from EN (source language) to multiple target languages (DE, ES, FR, IT, PT). The dataset incorporates a variety of gender phenomena, contains diverse sentence structures, covers morphologically different languages, and is gender balanced. 

### Bias
Bias can exist in macine translation since the vast amount of data that a translation model is trained on can contain inherent biases. One such example is gender bias. In this dataset, sentences to translate contain references to male and female gender identities. The performance of the model can then be evaluated using a custom version of BiLingual Evaluation Understudy (BLEU) that considers both overall translation performance and the performance gap for different gender identities.

### Pretrained Model, fine-tuning, and inference

Here we use the pretrained [google/flan-t5-xl model](https://huggingface.co/google/flan-t5-xl) (3B parameters) from the Hugging Face platform.

In the `machine-translation-t5-xl-pretrained` notebook, we directly use the pretrained model for inference. Compute resource: Amazon SageMaker notebook instance (ml.g4dn.xlarge).

In the `machine-translation-t5-xl-fine-tuning` notebook, we fine-tune the model first using our training dataset, and then use the fine-tuned model for inference. In addition, we also fine-tune the model with Deepspeed (reference: https://github.com/microsoft/DeepSpeed, https://huggingface.co/docs/accelerate/usage_guides/deepspeed). Compute resource: AWS EC2 instance (p5.48xlarge).

For details, please take a look at the notebooks.

### Environment setup for AWS EC2 p5 instance

1. Basic environment

- AMI: Deep Learning AMI GPU PyTorch 2.0.1 (Amazon Linux 2) 20231003
- SSM into the instance
- Set up Python virtual environment
```
sudo su - ec2-user
cd ~                                   # /home/ec2-user
mkdir p5-fine-tuning
cd p5-fine-tuning
python3 --version                      # Python 3.10.9
conda create --name ml python=3.10.9
conda init bash
source /home/ec2-user/.bashrc          # -> (base)
conda activate ml                      # -> (ml)
```
- Set up and run Jupyter Notebook
```
# in EC2 instance
pip install jupyter
python -m ipykernel install --user --name ml --display-name "ml"
# start the notebook
nohup jupyter notebook --no-browser --port=8000 &
jupyter notebook list  # confirm that the server is running

# in local machine
ssh -i <path/to/pem/file> -N -f -L localhost:8000:localhost:8000 ec2-user@<EC2 address>

# in browser
localhost:8000/?token=....
```

2. Set up Pytorch
Since p5 instances use NVIDIA H100 GPUs which have a compute capability of 9.0, you might get this warning later because of incorrect Pytorch version that does not support the GPU and CUDA version.

>NVIDIA H100 80GB HBM3 with CUDA capability sm_90 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86.
If you want to use the NVIDIA H100 80GB HBM3 GPU with PyTorch, please check the instructions athttps://pytorch.org/get-started/locally/


- Check the CUDA version
```
nvcc --version
# ... release 12.1 ...
```
- Uninstall and reinstall Pytorch
```
pip3 uninstall torch
pip3 install torch torchvision torchaudio

# confirm Pytorch version
python3 -c "import torch; print(torch.__version__)"
# 2.1.0+cu121
```


3. Set up Deepspeed
```
pip3 install -q deepspeed ninja --upgrade

# can double check the status using the Deepspeed report:
python -c "import deepspeed; print(deepspeed.__version__)" && ds_report
```

4. Set up NCCL (optional)
You might need to set up NCCL communicator for multi-GPU and/or multi-node environment

```
# confirm that all these directories already exist:
CUDA_DIRECTORY=/usr/local/cuda
EFA_DIRECTORY=/opt/amazon/efa
OPENMPI_DIRECTORY=/opt/amazon/openmpi

# these 2 directories don't exist yet:
NCCL_DIRECTORY=~/nccl
AWS_OFI_DIRECTORY=~/aws-ofi-nccl

# installation
cd ~
git clone https://github.com/NVIDIA/nccl.git
cd nccl
make -j src.build
cd ~
git clone https://github.com/aws/aws-ofi-nccl.git -b aws
cd aws-ofi-nccl
./autogen.sh
./configure --with-mpi=$OPENMPI_DIRECTORY --with-libfabric=$EFA_DIRECTORY --with-nccl=$NCCL_DIRECTORY/build --with-cuda=$CUDA_DIRECTORY
export PATH=$OPENMPI_DIRECTORY/bin:$PATH
make
sudo make install

# NCCL tests
cd ~
git clone https://github.com/NVIDIA/nccl-tests.git
cd  nccl-tests/
make NCCL_HOME=$CUDA_DIRECTORY
NCCL_DEBUG=INFO build/all_reduce_perf -b 8 -f 2 -e 32M -c 1
```
