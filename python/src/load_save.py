import numpy as np
import json, pickle
import scipy.io as spio


def conv_mat_data(mat_data_dir, py_data_dir, dataset):
	"""
	Convert a matlab data set to a python dictionary
	"""
	
	def check_keys(dict):
		for key in dict:
			if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
				dict[key] = _todict(dict[key])
		return dict        

	def _todict(matobj):
		dict = {}
		for strg in matobj._fieldnames:
			elem = matobj.__dict__[strg]
			if isinstance(elem, spio.matlab.mio5_params.mat_struct):
				dict[strg] = _todict(elem)
			else:
				dict[strg] = elem
		return dict
		
	file_in = r'%s\%s.mat' % (mat_data_dir, dataset)
	mat_data = spio.loadmat(file_in, struct_as_record=False, squeeze_me=True)
	data = check_keys(mat_data)['all_data']
	file_out = '%s/%s.pkl' % (py_data_dir, dataset)
	with open(file_out, 'wb') as f:
		pickle.dump(data, f)

def load_data(data_dir, dataset):
	"""
	Load python dictionary containing FRET data
	"""
	
	file = r'%s/%s.pkl' % (data_dir, dataset)
	with open(file, 'rb') as f:
		data = pickle.load(f)
	return data