### Metric Analysis

These tools provide a means of analyzing metric data obtained from profiling your program. For details on how to use nvprof
and the visual profiler, please see our in-depth guide. 

Once you have profiled your program and opened it with the visual profiler, you can then obtain csv files describing your 
metrics. To do so, press the "Export to CSV" button, as seen highlighted in red below. By default, the "GPU Details" tab 
lists only a summary of your data. To view a full list, press the button highlighted in purple. Again, you can export these 
details to another csv file.

![Single-Precision Function Unit Utilization](https://i.imgur.com/CcQANbE.png)

 Once you have obtained both files, you can analyze them with the two utilities in this directory. *Metric_analysis.py* is 
 to be used with the complete csv file, and *Summary_metric_analysis.py* is to be used with the summary csv file.
 
 ```bash
$ python Metric_analysis.py metrics_full.csv > full_output.log
$ python Summary_metric_analysis.py metrics_summary.csv > summary_output.log
```
