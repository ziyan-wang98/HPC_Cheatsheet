# HPC TF2.11.0 GPU

## 1. Install Miniconda / ANACONDA

## 2. Create a conda environment

1. IF you want python 3.8, then:
    
    ```bash
    conda create --name tf-py38 python=3.8
    ```
    
2. IF you want python 3.7 then:
    
    ```
    conda create --name tf-py37 python=3.7.4
    ```
    
    The reason is python 3.7 not support typing.py, "ImportError: cannot import name 'OrderedDict' from 'typing'" , so just do 3.7.4
    
    ```bash
    conda activate tf-py38 / tf-py37
    ```
    

## 3. GPU setup

1. • `pip install --upgrade setuptools pip`
2. • `pip install nvidia-pyindex`
3. • `pip install nvidia-tensorrt== 8.0+`  Just try it and check the version you can install.
4. verify by:
    
    ```bash
    python3 -c "import tensorrt; print(tensorrt.__version__); assert tensorrt.Builder(tensorrt.Logger())"
    ```
    
5. Go to your conda address: /env/tf-py37/lib/python3.7/site-packages/tensorrt/ and check the files you have, like:
    
    ![Untitled](HPC%20TF2%2011%200%20GPU%20dc576fb639a14603bc6654ab1016f5e5/Untitled.png)
    
    If you don’t have libnvinfer.so.7 and libnvinfer_plugin.so.7, then do:
    
    ```bash
    ln -s libnvinfer_plugin.so.8 libnvinfer_plugin.so.7
    ln -s libnvinfer.so.8 libnvinfer.so.7
    ```
    
6. go to your ~/.bashrc file, add this 
    
    ```bash
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/envs/tf-py37/lib/python3.7/site-packages/tensorrt/
    ```
    
7. Remember to source your bash file!!!!! 
    
    ```bash
    source ~/.bashrc
    
    conda activate tf-py37
    ```
    

## 4. To get started, install TensorFlow and follow the remaining steps

```bash
pip install tensorflow==2.11.0
```

**Do not use**: `conda install tensorflow` or `conda install tensorflow-gpu`

```bash
conda install nvidia-cuda-toolkit
```

Module need to load, depends on your hpc, try to spider first to make sure your hpc supply those:

```bash
module load gcc/9.4.0-gcc-9.4.0
module load mesa/21.2.3-gcc-9.4.0
module load llvm/12.0.1-gcc-9.4.0
module load cuda/11.4.2-gcc-9.4.0
module load cudnn/8.2.4.15-11.4-gcc-9.4.0
```

## 5. SLURM with conda

```bash
#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH -t 2-00:00:00
#SBATCH --cpus-per-task=68
#SBATCH -o /scratch/users/%u/sbatch_log/humanoid_our_wop/%j.out
#SBATCH --mem=500G

source ~/.bashrc

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate tf2

CUDA_VISIBLE_DEVICES=0,1

module load gcc/9.4.0-gcc-9.4.0
module load mesa/21.2.3-gcc-9.4.0
module load llvm/12.0.1-gcc-9.4.0
module load cuda/11.4.2-gcc-9.4.0
module load cudnn/8.2.4.15-11.4-gcc-9.4.0

for i in 0 1 2 3 4
do
    /users/k2257777/miniconda3/envs/tf2/bin/python /scratch/users/k2257777/FactoredRD/train.py \
    --env Humanoid-v2 --alg causal --apply-accurate-loss --full-structure-initial --policy-learning-with-causal\
    --zr-sparsity-coef 1e-5 --ar-sparsity-coef 1e-8 --temperature 1.0 --temperature-adjust-automatically\
    --zz-sparsity-coef 1e-5  --zz-sparsity-coef-aux 1e-7 --az-sparsity-coef 1e-8\
    --sparsity-loss-type 'cross_entropy'\
    --with-wandb --wandb-key c6b49827372958a9d636ecb5c024c16fe80c1bd0 &

done
wait
```

## Some problem:

1. ****Cannot load libcuda.so.1:****
    
    Use GPU code run in only cpu server. Try to run in GPU or check nvidia-smi the driver number is same as the:
    
    ```bash
    # See where the link is pointing.查看链接指向  
    ls  /usr/lib/x86_64-linux-gnu/libcuda.so.1 -la
    # My result:
    # lrwxrwxrwx 1 root root 19 Feb 22 20:40 \
    # /usr/lib/x86_64-linux-gnu/libcuda.so.1 -> ./libcuda.so.375.39
     
    # Make sure it is pointing to the right version. 查看NVIDIA驱动真正使用的版本
    # Compare it with the installed NVIDIA driver.
    nvidia-smi
     
    # Replace libcuda.so.1 with a link to the correct version，如果版本不对应的话，就要将链接重新对应上去
    cd /usr/lib/x86_64-linux-gnu
    sudo ln -f -s libcuda.so.<yournvidia.version> libcuda.so.1
    # 如我的NVIDIA版本是 384.130，命令行为
    #sudo ln -f -s libcuda.so.384.130 libcuda.so.1
    ```
    
    ![Untitled](HPC%20TF2%2011%200%20GPU%20dc576fb639a14603bc6654ab1016f5e5/Untitled%201.png)