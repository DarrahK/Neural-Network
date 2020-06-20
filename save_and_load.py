import pickle

def save(file_name, object):
	file = open(file_name, 'wb')
	pickle.dump(object, file)

def load(file_name):
	file = open(file_name, 'rb')
	return pickle.load(file)

