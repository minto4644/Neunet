def splitData(ratio):
	# Filtering out the variables that
	# have very less unique values using simple statistics.

	isCat = []
	drop_ = []
	count = 0
	for col in data.columns:
		if data[col].nunique() > 1:

			if 1. * data[col].nunique() / data[col].count() < 0.0001:
				isCat.append(col)
				uniq = data[col].unique()

				# print("\nThe number of unique\
				# 	values in Variable %s is %d" % (col, len(uniq)))
				# print(uniq)
				count += len(uniq)

		else:
			drop_.append(col)

	# Drop columns that have only one unique value.
	filtered_data = data.drop(drop_, axis=1)

	# Split the training dataset as
	# 75,15,10 % respectively for train,validation and test.

	# print("shape: %s" % str(data.shape))
	Y = filtered_data[target_variable]
	Y = Y.astype('str')
	X = filtered_data.drop([target_variable], axis=1)
	target_outputs = Y.unique()
	print("Target outputs: %s, len: %d" % (str(target_outputs), len(target_outputs)))

	X = X.astype('float64')
	X_mean = X.mean()
	X_std = X.std()

	for col in X.columns:
		X.loc[X[col].isnull(), col] = X_mean[col]

	X_norm = ((X - X_mean) / X_std)

	# Convert potential columns that seems Categorical using one-hot encoding.
	# X_norm = pd.get_dummies(X_norm,columns=isCat)

	# X_, self.X_test, y_, self.y_test = train_test_split(X_norm, Y,
	#	test_size=0.1, random_state=123)

	X_train, X_test, y_train, y_test = train_test_split(X_norm, Y, test_size = ratio, random_state = 145)
	lb = LabelBinarizer()
	# one_hot.fit(self.target_outputs)
	# print("y_train: %s" %(target_outputs[:2]))
	lb.fit(target_outputs)
	y_train_ = lb.transform(y_train)
	# print("one-hot lambda2ables: %s" %(one_hot.lables))
	print("lb classes : %s" % str(lb.classes_))
	# print("y_train_: %s " % str(y_train_[:2]))
	y_test_ = lb.transform(y_test)

	if len(target_outputs) == 2:
		y_train_ = np.hstack((1 - y_train_, y_train_))
		y_test_ = np.hstack((1 - y_test_, y_test_))
	# self.y_val_ = lb.transform(self.y_val)

	y_train_prob = pd.np.array(y_train_, dtype=pd.np.float64)
	y_test_prob = pd.np.array(y_test_, dtype=pd.np.float64)
	#print("y_train_prob: %s" %str(y_train_prob[:2]))
	# self.y_val_prob = pd.np.array(self.y_val_, dtype=pd.np.float64)

	return (X_train, X_test, y_train_prob, y_test_prob, target_outputs)