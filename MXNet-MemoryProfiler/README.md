# MXNet GPU Memory Profiler

This folder contains the git patch file which will modify MXNet source to generate annotations which can be used to create a memory profile of the mxnet/sockeye models. Further, the folder has scripts to analysis the mxnet/sockeye log file and plot the memory profile on a graph.

The profiler is based on a modified version of MXNet. Therefore installing the memory profiler will essentially require a reinstallation of MXNet. We provide two ways to install our profiler: install from scratch, and install with vertualenv. Our current profiler is based on the code base of MXNet version 0.12.0.

## Install MXNet With Memory Profiler From Scratch
- __Step 1__ The script 'patch_profiler.sh' downloads mxnet, switches to v0.12 and applies the profiler git .patch file 'memprofilerv12.patch'. To use it, change directory to folder containing 'patch_profiler.sh' and 'memprofilerv12.patch' and run 'patch_profiler.sh'.
```
cd MXNet-MemoryProfiler
./patch_profiler.sh
```

**Prerequisites**

Our memory profiler is based on MXNet with GPU support. Please install CUDA 9.0 and cuDNN 7.0 according to the following instructions:

- __Step 2__ Install CUDA 9.0 following the NVIDIA’s [installation guide.](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

- __Step 3__ Install cuDNN 7 for CUDA 9.0 following the NVIDIA’s [installation guide.](https://developer.nvidia.com/cudnn) You may need to register with NVIDIA for downloading the cuDNN library.

__Note:__ Make sure that the CUDA install path is added to LD_LIBRARY_PATH: \
`export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH`

__Note:__ Make sure that GCC version is 4.8 or later to compile C++ 11.

**Build the MXNet core shared library and python bindings**

- __Step 4__ Change the `MXNET_HOME` variable in `mxnet/setup-utils/install-mxnet-ubuntu-python.sh` to the current directory of mxnet.

- __Step 5__ Change the `USE_CUDA`, `USE_CUDA_PATH` and `USE_CUDNN` variables in `mxnet/make/config.mk`:
```
USE_CUDA=1
USE_CUDA_PATH=/usr/local/cuda
USE_CUDNN=1
```
Then copy the `config.mk` to `mxnet` root directory.

- __Step 6__ Install MXNet with python binding:
```
cd mxnet
bash setup-utils/install-mxnet-ubuntu-python.sh
```

**Validate MXNet Installation**

Start the python terminal.
```
$ python
```
Run a short MXNet python program to create a 2X3 matrix of ones a on a GPU, multiply each element in the matrix by 2 followed by adding 1. We expect the output to be a 2X3 matrix with all elements being 3. We use mx.gpu(), to set MXNet context to be GPUs.
```
>>> import mxnet as mx
>>> a = mx.nd.ones((2, 3), mx.gpu())
>>> b = a * 2 + 1
>>> b.asnumpy()
array([[ 3.,  3.,  3.],
       [ 3.,  3.,  3.]], dtype=float32)
```
**We have successfully installed the modified MXNet now we describe how to use it to generate memory profile graphs**

Note the following:
1. Refer [here](https://github.com/awslabs/sockeye) to understand how to install and use Sockeye.
2. Refer [here](https://github.com/apache/incubator-mxnet/tree/master/example) to see MXNet usage examples.
3. **The memory profiler provided here works with MXNET Version v0.12.0 hence it can be used with Sockeye versions that support MXNET v0.12.0.** This memory profiler has been tested with Sockeye version 1.12.3 of the branch 'origin/arxiv_1217'. When installing sockeye, You can checkout the archive branch by:
```
git checkout origin/arxiv_1217
```

## USAGE

1. Run whatever sockeye/mxnet model you want to profile and place the stderr output file in a folder. Let us call this folder 'logs'.
For example, save a sockeye models output to file in the following manner:
```
python3 -m sockeye.train \
-s /home/ab/clone/data/wmt17/corpus.tc.BPE.de \
-t /home/ab/clone/data/wmt17/corpus.tc.BPE.en \
-vs /home/ab/clone/data/wmt17/newstest2016.tc.BPE.de \
-vt /home/ab/clone/data/wmt17/newstest2016.tc.BPE.en \
--source-vocab /home/ab/sockeye/wmt_model/vocab.src.json \
--target-vocab /home/ab/sockeye/wmt_model/vocab.trg.json \
--num-embed 256 \
--rnn-num-hidden 512 \
--rnn-attention-type dot \
--use-tensorboard \
-o wmt_model \
--seed=1 \
--device-ids=-2 \
--batch-size 2 \
--bucket-width 10 \
--optimized-metric bleu \
--max-updates 1 > log_file 2>&1
```
Note how we redirect stderr and stdout output to 'log_file' above, the stderr output contains annotations from the memory profiler 
that we will use to generate the memory profile graph.

2. Use 'memory_analysis.py' script. Pass, as command line argument, the path of the directory containing the stderr log file of sockeye/mxnet (in our example: 'logs' folder from above step). The script will generate an analysis file (ending with 'ANALYSIS') for each log file in the folder that was supplied to the script. These 'ANALYSIS' files contain a json dump of information of all types of allocations found by the memory profiler. The generated analysis files will be placed in a folder called 'memory_analysis', in the current working directory.
```
python3 memory_analysis.py /path/to/logs
```

3. Next, to plot the corresponding graphs: use script 'plot_memory_analysis.py' and pass the path to 'memory_analysis' folder (generated in previous step) to this script. It will plot one graph for each 'ANALYSIS' file in 'memory_analysis' folder. It will place the graphs in 'memory_analysis_graphs' in current working directory. The script has other options such as to plot all files on same graph (used for comparison studies), explore the other options by running 'python3 plot_memory_analysis.py -h'.
```
python3 plot_memory_analysis.py memory_analysis
```

## QUICK SUMMARY
Use scripts to clone mxnet and patch memory profiler:
```
cd MXNet-MemoryProfiler
./patch_profiler.sh
```
Build MXNet **from source** by following the instructions here:

[MXNet Installation instructions.](https://mxnet.incubator.apache.org/install/index.html)

Now, run sockeye / mxnet models and lets say the stderr log file generated from these runs are in folder 'logs'. Generate memory profile graphs:
```
python3 memory_analysis.py /path/to/logs
python3 plot_memory_analysis.py memory_analysis
```
