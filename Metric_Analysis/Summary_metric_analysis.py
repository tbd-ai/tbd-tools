import csv
import sys

csv_name = sys.argv[1]
kernel_total_length = {}
kernel_weighted_util = {}

with open(csv_name, 'r') as csv_file:
	total_duration = 0
	total_weighted_util = 0
	reader = csv.DictReader(csv_file)

	for row in reader:
		name = row['Name']
		avg_duration = float(row['Avg. Duration(ns)'])
		invocations = float(row['Invocations'])
		duration = avg_duration * invocations
		util_str = row['Single-Precision Function Unit Utilization']
		util = 0
		if util_str == '':
			util = 0
			print(row)
			print('warning')
		elif util_str == 'Max':
			util = 10
		else:
			util = float(util_str[-2])

		if name not in kernel_total_length.keys():
			kernel_total_length[name] = duration
			kernel_weighted_util[name] = duration * util
		else:
			kernel_total_length[name] += duration
			kernel_weighted_util[name] += duration * util

		total_duration += duration
		total_weighted_util += duration * util

	print("total duration (ns)", total_duration)
	total_util = total_weighted_util / total_duration
	print("average util scale", total_util)
	print("==========================================")

	L1 = []
	for k in kernel_total_length.keys():
		L1.append(k)

	for i in range(len(L1)):
		for j in range(i, len(L1)):
			if kernel_total_length[L1[i]] < kernel_total_length[L1[j]]:
				L1[i], L1[j] = L1[j], L1[i]

	count = 0
	print("top 5 kernels with high utilization:")
	for i in range(len(L1)):
		if kernel_weighted_util[L1[i]] / kernel_total_length[L1[i]] > total_util:
			count += 1
			print(format(kernel_weighted_util[L1[i]] / kernel_total_length[L1[i]], '.2f'), format(kernel_total_length[L1[i]] / total_duration * 100, '.2f'), L1[i])
			if count == 5:
				break

	print("==========================================")
	count = 0
	print("top 5 kernels with low utilization:")
	for i in range(len(L1)):
		if kernel_weighted_util[L1[i]] / kernel_total_length[L1[i]] < total_util:
			count += 1
			print(format(kernel_weighted_util[L1[i]] / kernel_total_length[L1[i]], '.2f'), format(kernel_total_length[L1[i]] / total_duration * 100, '.2f'), L1[i])
			if count == 5:
				break

	#for k in kernel_total_length.keys():
	#	print(k, kernel_weighted_util[k] / kernel_total_length[k], kernel_total_length[k])
