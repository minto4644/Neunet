def load_data(fname):

	if fname is not '':
		print(fname)
		ext = fname.split('.')[-1]
		print("load_data")


		if ext == 'tsv':

			data = pd.read_csv(fname, sep ='\s+', prefix='X', low_memory=False, header=None)
			print(data[:2])
		else:
			data = pd.read_csv(fname, low_memory = False)

	return data