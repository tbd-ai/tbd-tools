# MXNET MEMORY PROFILER

This folder contains the git patch file which will modify MXNet source to generate annotations which can be used to create a memory profile of the mxnet/sockeye models. Further, the folder has scripts to analysis the mxnet/sockeye log file and plot the memory profile on a graph.

## USAGE
1. The script 'patch_profiler.sh' downloads mxnet, switches to v0.12 and applies the profiler git .patch file 'memprofilerv12.patch'. To use it, change directory to folder containing 'patch_profiler.sh' and 'memprofilerv12.patch' and run 'patch_profiler.sh'.
```
cd MXNet-MemoryProfiler
./patch_profiler.sh
```

2. Now edit 'mxnet/setup-utils/install-mxnet-ubuntu-python.sh'. At the start, change the line:
```
MXNET_HOME="$HOME/mxnet_old/"
```
to the path of mxnet we just cloned:
```
MXNET_HOME="/path/to/MXNet-MemoryProfiler/mxnet"
```

3. Run the following to install the updated mxnet:
```
bash setup-utils/install-mxnet-ubuntu-python.sh
```

4. Run whatever sockeye/mxnet model you want to profile and place the stderr output file in a folder. Let us call this folder 'logs'.
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

5. Use 'memory_analysis.py' script. Pass, as command line argument, the path of the directory containing the stderr log file of sockeye/mxnet (in our example: 'logs' folder from above step). The script will generate an analysis file (ending with 'ANALYSIS') for each log file in the folder that was supplied to the script. These 'ANALYSIS' files contain a json dump of information of all types of allocations found by the memory profiler. The generated analysis files will be placed in a folder called 'memory_analysis', in the current working directory.
```
python3 memory_analysis.py /path/to/logs
```

6. Next, to plot the corresponding graphs: use script 'plot_memory_analysis.py' and pass the path to 'memory_analysis' folder (generated in previous step) to this script. It will plot one graph for each 'ANALYSIS' file in 'memory_analysis' folder. It will place the graphs in 'memory_analysis_graphs' in current working directory. The script has other options such as to plot all files on same graph (used for comparison studies), explore the other options by running 'python3 plot_memory_analysis.py -h'.
```
python3 plot_memory_analysis.py memory_analysis
```

## QUICK USAGE SUMMARY
Use scripts to clone mxnet and patch memory profiler:
```
cd MXNet-MemoryProfiler
./patch_profiler.sh
```
Change directory to mxnet/setup utils and set MXNET_HOME correctly
```
cd mxnet/setup-utils
vim setup-utils/install-mxnet-ubuntu-python.sh
```
change :
```
MXNET_HOME="$HOME/mxnet_old/"
```
to:
```
MXNET_HOME="/path/to/MXNet-MemoryProfiler/mxnet"
```
Install mxnet:
```
bash install-mxnet-ubuntu-python.sh
```
Now, run sockeye / mxnet models and lets say the stderr log file generated from these runs are in folder 'logs'. Generate memory profile graphs:
```
python3 memory_analysis.py /path/to/logs
python3 plot_memory_analysis.py memory_analysis
```