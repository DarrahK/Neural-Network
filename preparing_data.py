
import numpy as np
import csv

# This will only work for CV

def load_data(file_name, num_of_inputs):

	with open(file_name) as csv_file:
		
		csv_reader = csv.reader(csv_file, delimiter=',')
		data = []

		for line in csv_reader:
			input = np.array([line[:num_of_inputs - 1]])
			input.transpose()

			output = np.array([line[num_of_inputs:]])
			output.transpose()

			data.append((np.array(input), np.array(output)))

	return data

