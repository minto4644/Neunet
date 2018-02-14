def trainModel():


	x = tf.placeholder(tf.float64, shape=[None, X_train.shape[1]], name='x')
	print("Len of target outputs: %d " % len(target_outputs))
	y_true = tf.placeholder(tf.float64, shape=[None, len(target_outputs)], name='y_true')

	# Let us build a simple two layer neural network
	# self.layers = hidden_layers
	hidden_layers.insert(0, X_train.shape[1])
	hidden_layers.append(len(target_outputs))
	W1 = tf.Variable(tf.truncated_normal(
		shape = [hidden_layers[0], hidden_layers[1]],
		stddev = 0.05, dtype = tf.float64))

	W2 = tf.Variable(tf.truncated_normal(shape=[hidden_layers[1], hidden_layers[2]], stddev=0.05, dtype=tf.float64))
	W3 = tf.Variable(tf.truncated_normal(shape=[hidden_layers[2], hidden_layers[3]], stddev=0.05, dtype=tf.float64))
	W4 = tf.Variable(tf.truncated_normal(shape=[hidden_layers[3],hidden_layers[4]],stddev=0.05,dtype=tf.float64))
	W5 = tf.Variable(tf.truncated_normal(shape=[hidden_layers[4],hidden_layers[5]],stddev=0.05,dtype=tf.float64))
	W6 = tf.Variable(tf.truncated_normal(shape=[hidden_layers[5],hidden_layers[6]],stddev=0.05,dtype=tf.float64))

	weights = tf.trainable_variables()

	b1 = tf.Variable(tf.truncated_normal(shape = [hidden_layers[1]], stddev = 0.01, dtype = tf.float64))
	b2 = tf.Variable(tf.truncated_normal(shape = [hidden_layers[2]],stddev = 0.01, dtype = tf.float64))
	b3 = tf.Variable(tf.truncated_normal(shape=[hidden_layers[3]],stddev=0.01,dtype=tf.float64))
	b4 = tf.Variable(tf.truncated_normal(shape=[
	hidden_layers[4]],stddev=0.01,dtype=tf.float64))
	b5 = tf.Variable(tf.truncated_normal(shape=[hidden_layers[5]],
		stddev=0.01, dtype=tf.float64))
	b6 = tf.Variable(tf.truncated_normal(shape=[hidden_layers[6]],
		stddev=0.01, dtype=tf.float64))

	out1 = tf.matmul(x, W1) + b1
	if activation_layer1.split('.')[-1] != 'linear':
		# act1 = actv_fn(activtion_layer1)
		out1 = eval(activation_layer1)(out1)
	# out1 = tf.sigmoid(out1)
	if dropout_btn:
		out1 = tf.nn.dropout(out1, dropout)

	out2 = tf.matmul(out1, W2) + b2
	if activation_layer2.split('.')[-1] != 'linear':
		# act2 = actv_fn(activtion_layer2)
		out2 = eval(activation_layer2)(out2)
	# out2 = tf.sigmoid(out2)
	if dropout_btn:
		out2 = tf.nn.dropout(out2, dropout)

	out3 = tf.matmul(out2, W3) + b3
	if activation_layer3.split('.')[-1] != 'linear':
		# act3 = actv_fn(activtion_layer3)
		out3 = eval(activation_layer3)(out3)
	if dropout_btn:
		out3 = tf.nn.dropout(out3, dropout)

	out4 = tf.matmul(out3, W4) + b4
	if activation_layer4.split('.')[-1] != 'linear':
		# act4 = actv_fn(activtion_layer4)
		out4 = eval(activation_layer4)(out4)
	if dropout_btn:
		out4 = tf.nn.dropout(out4, dropout)

	out5 = tf.matmul(out4, W5) + b5
	if activation_layer5.split('.')[-1] != 'linear':
		# act5 = actv_fn(activtion_layer5)
		out5 = eval(activation_layer5)(out5)
	print(dropout_btn)
	if dropout_btn:
		out5 = tf.nn.dropout(out5, dropout)

	out6 = tf.matmul(out5, W6) + b6
	# class_weight = tf.Variable([0.1,0.9],dtype=tf.float64)
	weighted_logits = out6

	y_pred = tf.nn.softmax(weighted_logits)
	y_pred_cls = tf.argmax(y_pred,dimension=1)
	y_true_cls = tf.arg_max(y_true,dimension=1)

	# print(tf.trainable_variables())
	# print("logits size: %s , labels size: %s" %(weighted_logits.shape(), y_true.shape()))
	loss = tf.nn.softmax_cross_entropy_with_logits(logits=weighted_logits,labels=y_true)
	regularizer1 = tf.nn.l2_loss(W1)
	regularizer2 = tf.nn.l2_loss(W2)
	regularizer3 = tf.nn.l2_loss(W3)
	regularizer4 = tf.nn.l2_loss(W4)
	regularizer5 = tf.nn.l2_loss(W5)
	regularizer6 = tf.nn.l2_loss(W6)




	loss1 = tf.nn.weighted_cross_entropy_with_logits(
		weighted_logits, y_true, pos_weight = 0.99, name = 'loss1')
	reg = 0
	if l1_regularization:
		print("l1 activated")
		
		reg += l1_regularization*(regularizer6 + regularizer5 + regularizer4 + regularizer3 + regularizer2 + regularizer1)
	if l2_regularization:
		print("L2 activated")
		l1_reg = tf.contrib.layers.l1_regularizer(scale=l2_regularization, scope=None)
		reg += tf.contrib.layers.apply_regularization(l1_reg, weights)
	cost = tf.reduce_mean(loss + reg)



	optimizer = optimizer_func.minimize(cost)

	sess = tf.Session()
	init = tf.initialize_all_variables()

	sess.run(init)
	feed_dict_train = {x: X_train.values, y_true: y_train_prob}
	feed_dict_test = {x: X_test.values, y_true: y_test_prob}
	# feed_dict_val = {x:self.X_val.values,y_true: self.y_val_prob}

	# from sklearn.metrics import confusion_matrix,f1_score,precision_recall_fscore_support,roc_curve,roc_auc_score
	# cof,y_true_,y_pred_,y_pred_prob = sess.run([tf.confusion_matrix(y_true_cls,y_pred_cls),y_true_cls,y_pred_cls,y_pred],feed_dict_test)
	epochs = 20

	for i in range(400):
		_,co = sess.run([optimizer,cost],feed_dict=feed_dict_train)
		if(i%epochs ==0):
			print("Iteration: %d with loss: %f" %(i,co*10000))
			cof,y_true_,y_pred_,y_pred_prob = sess.run([tf.confusion_matrix(y_true_cls,y_pred_cls),y_true_cls,y_pred_cls,y_pred],feed_dict_test)
			print("Confusion Matrix:\n %s" % confusion_matrix(y_true_,y_pred_))

	eval_results = precision_recall_fscore_support(y_true_, y_pred_)
	
	cof,y_true_,y_pred_,y_pred_prob = sess.run([tf.confusion_matrix(y_true_cls,y_pred_cls),y_true_cls,y_pred_cls,y_pred],feed_dict_test)

	test_results = accuracy_score(y_true_,y_pred_)
	print(test_results)
	f1_sco_test = test_results

	cof,y_true_,y_pred_,y_pred_prob = sess.run([tf.confusion_matrix(y_true_cls,y_pred_cls),y_true_cls,y_pred_cls,y_pred],feed_dict_train)
	train_results = accuracy_score(y_true_,y_pred_)