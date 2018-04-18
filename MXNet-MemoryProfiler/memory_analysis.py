'''
	@author: atiwari@cs.toronto.edu
	
Script to parse the sockeye output with MXNET memory profile annotations. It generates a detailed
breakdown of what parts of the model consume how much memory.
'''
import argparse as ap
import numpy as np
import sys
import os
import json

parser = ap.ArgumentParser()
parser.add_argument('directory', help='path of directory containing the annotated files', type=str)
args = parser.parse_args()
files = [args.directory+'/'+file for file in os.listdir(args.directory)]

# TAKE EXTREME CARE TO ENSURE THAT REGEX MATCHES ARE AS 
# DISJOINT AS POSSIBLE, OTHER WISE ANALYSIS
# RESULTS WILL BE INCORRECT
# eg: use regex: '(target)' and not 'target', because the
# latter will match certain weights and gradients as well.

# ADD REGEXS IN ORDER OF PRECEDENCE TO GET CORRECT ANALYSIS:
# eg: the memory annotation tag: '(in_arg:source_target_embed_weight).' 
# will match both 'embed' and 'in_arg' but we want it
# to be counted in 'in_arg' only so 'in_arg' is placed 
# before 'embed' in regex_list

regex_list = [
# weights and gradients; The following tags must be placed before 'embed'
'in_arg', 'arg_grad', 
# optimizer state, allocated by python bindings for weight update; must be placed before embed
'optimizer',
# must be placed before dot
'workspace',
# rnn
'rnn', 'embed', ':_mul', 'rsqrt',':mean', 'att', ':split',
'logit', 'swapaxes', 'square', 'softmax', 'sequencereverse',
'dot', ':broadcast', 'zeros',
# conv
'sum', 'transpose', ':dropout', 'slice', 'cnn', 'arange', 'dot',
'fullyconnected', 'sequencemask',
# transf
'activation', 'reshape', 'transformer',
# misc
'_equal_scalar', '(source)', '(target)', '(target_label)',
'untagged', 'warning!,ctx_source_unclear'
]

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

MB = 1024*1024

def print_summary(stats_dict):
	# sort in descending order of allocation size of each category
	keys_in_order = sorted(stats_dict, key= lambda k: stats_dict[k][0], reverse=True)
	total_items = 0
	total_alloc = 0
	total_wts = 0
	total_fwd = 0
	total_unknown = 0
	print('**********************************')
	print('**********************************')
	print("Statistics for : ", file)
	print('**********************************')
	print('**********************************')
	for key in keys_in_order:
		print(regex_to_category_name_encoding[key], '=',
			  stats_dict[key][0]/MB, 'MBs',
			  'Count:#',
			  len(stats_dict[key][1]))
		total_items+=len(stats_dict[key][1])
		total_alloc+=stats_dict[key][0]/MB
		if key == 'in_arg' or key == 'arg_grad':
			total_wts+=stats_dict[key][0]/MB
		elif key == 'untagged' or key == 'warning!,ctx_source_unclear':
			total_unknown+=stats_dict[key][0]/MB
		else:
			total_fwd+=stats_dict[key][0]/MB
	print('#############  SUMMARY #############')
	print ('Total Allocations= ', total_alloc, 'MB')
	print ('Total Weights= ', total_wts, 'MB')
	print ('Total Forward tags= ', total_fwd, 'MB')
	print ('Total Unknown tags= ', total_unknown, 'MB')
	# SANITY CHECK : This should match $ grep 'Allocate' annotations_file;
	print ('Total Annotations= ', total_items)

# Check that all regexex have been allotted a Category name
for regex in regex_list:
	if regex not in regex_to_category_name_encoding.keys():
		print('The regex: ', regex, 'has not been allotted a Category Name.')
		regex_to_category_name_encoding[regex] = regex

# Create sub directory to store analysis files
analysis_dir='./memory_analysis'
if not os.path.exists(analysis_dir):
	os.makedirs(analysis_dir)

for file in files:
	# This dict has form dict = { 'tag' : [float: total size of allocations for this tag,
	# list of all annotations with this tag]}
	# eg: rnn_stats_dict = 
	# {'zeros': [36.0, ['(forward_features:_zeros1).\n']] }
	stats_dict = {}
	reader = open(file)
	for line in reader:
		if 'Allocate' in line:
			words=line.split(' ')
			matched = 0
			for regex in regex_list:
				if regex in words[6]:
					matched=1
					if regex in stats_dict.keys():
						#update total allocation for this category
						stats_dict[regex][0]+=float(words[2]) 
						#append specific annotation to list of annotations that matched this regex
						stats_dict[regex][1].append(words[6])
					else:
						# Initailize; Save [allocation size, list of annotation]
						stats_dict[regex] = [ float(words[2]), [words[6]] ]
					break
			if matched == 0:
				print('UNKNOWN TAG:', words[6], '---did not match any tag. Update regexes list!')
	reader.close()
	filename = analysis_dir + '/' + file.split('/')[len(file.split('/'))-1] + '_ANALYSIS'
	# save analysis to file
	with open(filename, 'w') as output_file:
		output_file.write(json.dumps(stats_dict, indent=2))
	# print summary
	print_summary(stats_dict)
	print (file, 'done')
