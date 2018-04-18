'''
	@author: atiwari@cs.toronto.edu
'''
import json
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt 
import numpy as np
import sys
import argparse as ap
import os

parser = ap.ArgumentParser()
parser.add_argument('directory', help='path of directory containing the output of memory analysis script. These output files names must end with _ANALYSIS' , type=str)
parser.add_argument('-t', '--title', help='subtitle for graphs', type=str)
parser.add_argument('-f', '--fraction', help="omit categories that consume less than FRACTION of total memory (default=0; don't omit anything)", type=float)
parser.add_argument('-s', '--setting', help="all- to plot all graphs in the same file (default=None, plots all graphs in individual files)", type=str)
args = parser.parse_args()

folder = args.directory
if args.title is None:
	title_string = ''
else: title_string = args.title
if args.fraction is None:
	fraction=0
else:
	fraction = args.fraction
files = [args.directory+'/'+file for file in os.listdir(args.directory) if file.endswith('_ANALYSIS')]

# Create sub directory to store graphs
analysis_dir='./memory_analysis_graphs'
if not os.path.exists(analysis_dir):
	os.makedirs(analysis_dir)

# CONSTANTS
MB = 1024*1024

# For pretty printing later
regex_to_category_name_encoding = {
	'rnn' : 'RNN Layer',
	'embed'	: 'Embedding Layer',
	':_mul' : 'Multiplication Op',
	'rsqrt' : 'Sqrt Op',
	':mean' : 'Mean Op',
	'att' : 'Attention Layer',
	':split' : 'Split Op',
	'split' : 'Split Op',
	'logit' : 'Logit',
	'swapaxes' : 'SwapAxes Op',
	'square' : 'Square Op',
	'softmax' : 'SoftMax',
	'sequencereverse': 'SequenceReverse Op',
	'dot' : 'Dot Op',
	':broadcast' : 'Broadcast Op',
	'zeros' : 'Zero Op',
	'sum' : 'Sum Op',
	'transpose' : 'Transpose Op',
	'dropout' : 'Dropout',
	':dropout' : 'Dropout',
	'slice' : 'Slice Op',
	'cnn' : 'CNN Layer',
	'arange' : 'Arange Op',
	'fullyconnected' : 'FullyConnected',
	'sequencemask' : 'SequenceMask',
	'activation' : 'Activation',
	'reshape' : 'Reshape Op',
	'transformer' : 'Transformer Layer',
	'in_arg' : 'Weights',
	'arg_grad' : 'Weight Gradients',
	'Weights' : 'Weights And Gradients',
	'optimizer' : 'Optimizer State',
	'_equal_scalar' : 'Equal Scalar Op',
	'(source)' : 'Source',
	'(target)' : 'Target',
	'(target_label)': 'Target Label',
	'workspace' : 'Workspace',
	'untagged' : 'Unknown',
	'warning!,ctx_source_unclear' : 'Unknown'
}

def get_model_name(file):
	if 'rnn' in file.lower():
		return 'RNN'
	elif 'conv' in file.lower():
		return 'CONVOLUTIONAL'
	elif 'tran' in file.lower():
		return 'TRANSFORMER'
	else: return file.lower()

def plot_all_on_same_graph(files):
	# Sortingso that graphs appear based on some order, assuming filenames are meaningful
	# eg: rnn_1g_2, rnn_1g_4, rnn_1g_8
	files.sort()
	from math import ceil
	squares = ceil(len(files)**(0.5))
	fig, axes = plt.subplots(nrows=squares, ncols=squares, figsize=(60,60))
	_fontsize = 40
	plt.subplots_adjust(top=0.9, bottom=0.2, left=0.10, right=0.95, hspace=0.9,
                    	wspace=0.1)
	plt.xlabel('Categories', fontsize=_fontsize)
	plt.ylabel('Memory in MBs', fontsize=_fontsize)
	index = 0

	for ax in axes.flat[:]:
		if index == len(files): break
		file = files[index]
		with open(file) as reader:
			stats_dict = json.load(reader)

		# Compute total allocation and maximum allocation
		total_alloc = 0
		max_alloc = 0
		for key in stats_dict.keys():
			total_alloc+=stats_dict[key][0]/MB
			if stats_dict[key][0]/MB > max_alloc:
				max_alloc = stats_dict[key][0]/MB

		# Set ylimit
		ax.set_ylim([0,max_alloc+500])

		# Plot all in same order
		keys_in_order = sorted(stats_dict.keys())
		print(keys_in_order)
		pos = 0
		keys_to_plot = []
		for key in keys_in_order:
			alloc = stats_dict[key][0]/MB
			fraction_of_total = (alloc/total_alloc)*100
			if fraction and fraction_of_total <= fraction:
				print('File:', file, "Omitting:", regex_to_category_name_encoding[key])
				continue
			pos+=1
			keys_to_plot.append(key)
			bar_conatiner = ax.bar(pos, alloc)
			bar = bar_conatiner.patches[0]

			ax.text(bar.get_x() + bar.get_width()/2.,
				    1.05*bar.get_height(),
				    "{0:.2f}%".format((alloc/total_alloc)*100),
				    ha='center',
				    va='bottom',
				    fontsize=_fontsize,
				    rotation=70)
		plt.sca(ax)
		plt.xticks(np.arange(1,len(keys_to_plot)+1),
				   [regex_to_category_name_encoding[key] for key in keys_to_plot])
		# plt.setp(ax.get_xticklabels(), rotation=90, fontsize=_fontsize)
		for tick in ax.get_xticklabels():
			tick.set_rotation(90)
			tick.set_fontsize(_fontsize)
		for tick in ax.get_yticklabels():
			tick.set_fontsize(_fontsize)

		# Printing file name as model name because in the use case where this function is called,
		# the filename contains meaningful information like model name, parameter etc. This is because
		# this function is used to plot different files on the same graph, these different files
		# are presumably the models run with different hyperparmas and hopefully the difference are
		# captured by the filenames.
		ax.set_title('\nModel:'+ file +\
			         '\nTotal Allocation:{0:.2f}MB'.format(total_alloc),
			         fontsize=_fontsize)
		# ax.set_xlabel('Categories', fontsize=_fontsize)
		# ax.set_ylabel('Memory in MBs', fontsize=_fontsize)
		index+=1
	plt.savefig(analysis_dir + '/memory_profile_graphs.pdf')

def plot_in_separate_files(files):
	figure=0
	for file in files:
		with open(file) as reader:
			stats_dict = json.load(reader)

		# Compute total allocation and maximum allocation
		total_alloc = 0
		max_alloc = 0
		for key in stats_dict.keys():
			total_alloc+=stats_dict[key][0]/MB
			if stats_dict[key][0]/MB > max_alloc:
				max_alloc = stats_dict[key][0]/MB

		# Sort in descending order of allocation size for each category
		keys_in_order = sorted(stats_dict, key= lambda k: stats_dict[k][0], reverse=True)

		figure += 1
		index = 0

		plt.figure(figure)
		fig, ax = plt.subplots()
		fig.subplots_adjust(bottom=0.4, top=0.8)

		# Set ylimit
		ax.set_ylim([0,max_alloc+500])

		# ax.text
		keys_to_plot = []
		for key in keys_in_order:
			index+=1
			alloc = stats_dict[key][0]/MB
			fraction_of_total = (alloc/total_alloc)*100
			if fraction and fraction_of_total <= fraction:
				print('File:', file, "Omitting:", regex_to_category_name_encoding[key])
				continue
			keys_to_plot.append(key)
			bar_conatiner = plt.bar(index, alloc)
			bar = bar_conatiner.patches[0]

			ax.text(bar.get_x() + bar.get_width()/2.,
				    1.05*bar.get_height(),
				    "{0:.2f}%".format((alloc/total_alloc)*100),
				    ha='center',
				    va='bottom',
				    fontsize=8,
				    rotation=70)
		plt.xticks(np.arange(1,len(keys_to_plot)+1),
				   [regex_to_category_name_encoding[key] for key in keys_to_plot],
				   rotation=90)
		plt.title('Memory Profile'+ \
				  '\n' + title_string + \
			      '\nModel:'+ get_model_name(file) +\
			      '\nTotal Allocation:{0:.2f}MB'.format(total_alloc)
			      )
		plt.xlabel('Categories')
		plt.ylabel('Memory in MBs')
		# plt.show()
		filename = analysis_dir + '/' + file.split('/')[len(file.split('/'))-1] + '_mem_graph.pdf'
		plt.savefig(filename)

if args.setting == 'all':
	plot_all_on_same_graph(files)
else:
	plot_in_separate_files(files)