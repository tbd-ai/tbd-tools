# MXNET MEMORY PROFILER

This folder contains the git patch file which will modify MXNet source to generate annotations which can be used to create a memory profile of the mxnet/sockeye models. Further, the folder has scripts to analysis the mxnet/sockeye log file and plot the memory profile on a graph.

## USAGE
The script 'patch_profiler.sh' downloads mxnet, switches to v0.12 and applies the profiler git .patch file 'memprofilerv12.patch'. To use it, change directory to folder containing 'patch_profiler.sh' and 'memprofilerv12.patch' and run 'patch_profiler.sh'.

To run the memory analysis:
1. Run whatever sockeye/mxnet model you want to profile and place the generated log file a folder. Let us call this folder 'logs'.

2. Use 'memory_analysis.py' script. Pass, as command line argument, the path of the directory containing the log file of sockeye/mxnet (in our example: 'logs' folder from step 1). The script will generate an analysis file (ending with 'ANALYSIS') for each log file in the folder that was supplied to the script. These 'ANALYSIS' files contain a json dump of information of all types of allocations found by the memory profiler. The generated analysis files will be placed in a folder called 'memory_analysis', in the current working directory.

3. Next, to plot the corresponding graphs: use script 'plot_memory_analysis.py' and pass the path to 'memory_analysis' folder (generated in previous step) to this script. It will plot one graph for each 'ANALYSIS' file in 'memory_analysis' folder. It will place the graphs in 'memory_analysis_graphs' in current working directory. The script has other options such as to plot all files on same graph (used for comparison studies), explore the other options by running 'python3 plot_memory_analysis.py -h'.

## QUICK USAGE SUMMARY
Clone mxnet and patch memory profiler

./patch_profiler.sh

run sockeye / mxnet model and lets say the log files generated from these runs are in folder 'logs', run memory analysis:

python3 memory_analysis.py /path/to/logs

The above will create folder 'memory_analysis' in the current working directory, plot the graphs as follows:

python3 plot_memory_analysis.py memory_analysis

The above will plot and place the graphs in 'memory_analysis_graphs' in current working directory
