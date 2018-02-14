"""

PyQt5 Tutorial

This example tries to read a
csv file using QFileDialog.

Author: Minto
Website: github/minto4644
Last Edited: 9th August 2017

"""

import sys
import time
import traceback
import subprocess
import inspect
from workersignals import WorkerSignals

# import time
import pandas as pd
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import LabelBinarizer


class Worker(QRunnable):
	'''
	A worker class that runs any function passed to it on background thread.
	'''
	def __init__(self, fn, *args, **kwargs):
		super().__init__()
		self.fn = fn
		self.args = args
		self.kwargs = kwargs
		self.signals = WorkerSignals()

	@pyqtSlot()
	def run(self):
		try:
			result = self.fn(*self.args, **self.kwargs)

		except:
			traceback.print_exc()
			exctype, value = sys.exc_info()[:2]
			self.signals.error.emit((exctype, traceback.format_exc()))
		else:
			self.signals.result.emit(result)
		finally:
			self.signals.finished.emit()



class Example(QMainWindow):

	def __init__(self):
		self.threadpool = QThreadPool()
		self.flag = 0
		self.counter = 0
		self.optimizer_func = ''
		self.args = []
		self.num_layers = 5
		self.hidden_layers = []
		super().__init__()

		self.initUI()

	def initUI(self):
		self.load_btn = QPushButton('Load data', self)
		self.load_btn.setGeometry(300, 300, 250, 30)
		self.load_btn.move(300, 40)
		self.load_btn.setStyleSheet("background-color: rgb(210, 221, 242)")
		self.load_btn.clicked.connect(self.showDialog)
		self.statusBar()

		# self.progressbar = QProgressBar(self)
		# self.progressbar.setGeometry(300, 300, 170, 30)
		# self.progressbar.move(50, 380)
		# self.progressbar.setValue(0)
		# self.progressbar.setStatusTip("Hi")


		self.select_target_variable = QComboBox(self)
		self.select_target_variable.setGeometry(300, 300, 250, 30)
		self.select_target_variable.move(300, 75)
		self.select_target_variable.setEnabled(False)
		self.select_target_variable.addItem('Select Target Variable')
		self.select_target_variable.setStyleSheet("background-color: rgb(210, 221, 242)")
		self.select_target_variable.currentIndexChanged.connect(self.selectionchange)

		self.split_btn = QPushButton('Split data', self)
		self.split_btn.setGeometry(350, 350, 200, 30)
		self.split_btn.move(350, 110)
		self.split_btn.clicked.connect(self.splitData)
		self.split_btn.setStyleSheet("background-color: rgb(210, 221, 242)")
		self.split_btn.setEnabled(False)

		self.ratio = QComboBox(self)
		self.ratio.setGeometry(350, 350, 80, 30)
		self.ratio.move(300, 110)
		self.ratio.setStyleSheet("background-color: rgb(247, 202, 203)")
		self.ratio.addItem("ratio")
		self.ratio.setEnabled(False)


		# self.trainData_btn = QPushButton('Train data', self)
		# self.trainData_btn.setGeometry(350, 350, 120, 30)
		# self.trainData_btn.move(300, 145)
		# self.trainData_btn.clicked.connect(self.selectTrain)
		# self.trainData_btn.setEnabled(False)

		# self.testData_btn = QPushButton('Test data', self)
		# self.testData_btn.setGeometry(350, 350, 120, 30)
		# self.testData_btn.move(430, 145)
		# self.testData_btn.clicked.connect(self.selectTest)
		# self.testData_btn.setEnabled(False)

		self.layer1_btn = QPushButton('Layer 1', self)
		self.layer1_btn.setGeometry(350, 350, 60, 30)
		self.layer1_btn.move(100, 230)
		self.layer1_btn.setEnabled(False)

		self.layer1_slider = QSpinBox(self)
		self.layer1_slider.setGeometry(350, 350, 40, 30)
		self.layer1_slider.move(160, 230)
		self.layer1_slider.setEnabled(False)

		self.activtion_layer1 = QComboBox(self)
		self.activtion_layer1.setGeometry(350, 350, 80, 30)
		self.activtion_layer1.move(200, 230)
		self.activtion_layer1.setEnabled(False)
		self.activtion_layer1.addItems(['linear','sigmoid','relu','tanh'])

		self.regularization_text = QLabel(self)
		self.regularization_text.setGeometry(350, 350, 30, 30)
		self.regularization_text.move(300, 230)
		self.regularization_text.setText('Reg')

		# self.lambda1_text = QLabel(self)
		# self.lambda1_text.setGeometry(350, 350, 10,30)
		# self.lambda1_text.move(350, 200)
		# self.lambda1_text.setText('l1')

		self.l1_regularization = QPushButton(self)
		self.l1_regularization.setGeometry(350, 350, 20, 30)
		self.l1_regularization.move(350, 230)
		self.l1_regularization.setText('L1')
		self.l1_regularization.setEnabled(False)
		self.l1_regularization.setCheckable(True)
		self.l1_regularization.setChecked(False)
		self.l1_regularization.clicked.connect(self.toggle1)

		self.lambda1 = QDoubleSpinBox(self)
		self.lambda1.setGeometry(350, 350, 70, 30)
		self.lambda1.move(370, 230)
		self.lambda1.setDecimals(8)
		self.lambda1.setRange(0.00000001, 5)
		self.lambda1.setEnabled(False)

		self.l2_regularization = QPushButton(self)
		self.l2_regularization.setGeometry(350, 350, 20, 30)
		self.l2_regularization.move(440, 230)
		self.l2_regularization.setText('L2')
		self.l2_regularization.setEnabled(False)
		self.l2_regularization.setCheckable(True)
		self.l2_regularization.setChecked(False)
		self.l2_regularization.clicked.connect(self.toggle2)

		self.lambda2 = QDoubleSpinBox(self)
		self.lambda2.setGeometry(350, 350, 70, 30)
		self.lambda2.move(460, 230)
		self.lambda2.setDecimals(8)
		self.lambda2.setRange(0.00000001, 5)
		self.lambda2.setEnabled(False)


		self.dropout_text = QLabel(self)
		self.dropout_text.setGeometry(350, 350, 60, 30)
		self.dropout_text.move(300, 265)
		# self.dropout_text.setTextInteractionFlags(Qt.TextInteractionFlag(0))
		self.dropout_text.setText("Dropout")

		self.dropout = QDoubleSpinBox(self)
		self.dropout.setGeometry(350, 350, 90, 30)
		self.dropout.move(390, 265)
		self.dropout.setRange(0.01, 1)
		self.dropout.setEnabled(False)

		self.dropout_btn = QPushButton(self)
		self.dropout_btn.setGeometry(350, 350, 20, 30)
		self.dropout_btn.move(350, 265)
		self.dropout_btn.setText('dr')
		self.dropout_btn.setEnabled(False)
		self.dropout_btn.setCheckable(True)
		self.dropout_btn.setChecked(False)
		self.dropout_btn.clicked.connect(self.toggle3)

		# self.learning_rate_text = QLabel(self)
		# self.learning_rate_text.setGeometry(350, 350, 30, 30)
		# self.learning_rate_text.move(440, 265)
		# self.learning_rate_text.setText('lr')

		# self.learning_rate = QDoubleSpinBox(self)
		# self.learning_rate.setGeometry(350, 350, 70, 30)
		# self.learning_rate.move(460, 265)
		# self.learning_rate.setEnabled(False)
		# self.learning_rate.setDecimals(4)

		self.layer2_btn = QPushButton('Layer 2', self)
		self.layer2_btn.setGeometry(350, 350, 60, 30)
		self.layer2_btn.move(100, 265)
		self.layer2_btn.setEnabled(False)

		self.layer2_slider = QSpinBox(self)
		self.layer2_slider.setGeometry(350, 350, 40, 30)
		self.layer2_slider.move(160, 265)
		self.layer2_slider.setEnabled(False)

		self.activtion_layer2 = QComboBox(self)
		self.activtion_layer2.setGeometry(350, 350, 80, 30)
		self.activtion_layer2.move(200, 265)
		self.activtion_layer2.setEnabled(False)
		self.activtion_layer2.addItems(['linear', 'sigmoid', 'relu', 'tanh'])

		self.layer3_btn = QPushButton('Layer 3', self)
		self.layer3_btn.setGeometry(350, 350, 60, 30)
		self.layer3_btn.move(100, 300)
		self.layer3_btn.setEnabled(False)

		self.layer3_slider = QSpinBox(self)
		self.layer3_slider.setGeometry(350, 350, 40, 30)
		self.layer3_slider.move(160, 300)
		self.layer3_slider.setEnabled(False)

		self.activtion_layer3 = QComboBox(self)
		self.activtion_layer3.setGeometry(350, 350, 80, 30)
		self.activtion_layer3.move(200, 300)
		self.activtion_layer3.setEnabled(False)
		self.activtion_layer3.addItems(['linear', 'sigmoid', 'relu', 'tanh'])

		self.layer4_btn = QPushButton('Layer 4', self)
		self.layer4_btn.setGeometry(350, 350, 60, 30)
		self.layer4_btn.move(100, 335)
		self.layer4_btn.setEnabled(False)

		self.layer4_slider = QSpinBox(self)
		self.layer4_slider.setGeometry(350, 350, 40, 30)
		self.layer4_slider.move(160, 335)
		self.layer4_slider.setEnabled(False)

		self.activtion_layer4 = QComboBox(self)
		self.activtion_layer4.setGeometry(350, 350, 80, 30)
		self.activtion_layer4.move(200, 335)
		self.activtion_layer4.setEnabled(False)
		self.activtion_layer4.addItems(['linear', 'sigmoid', 'relu', 'tanh'])

		self.layer5_btn = QPushButton('Layer 5', self)
		self.layer5_btn.setGeometry(350, 350, 60, 30)
		self.layer5_btn.move(100, 370)
		self.layer5_btn.setEnabled(False)

		self.layer5_slider = QSpinBox(self)
		self.layer5_slider.setGeometry(350, 350, 40, 30)
		self.layer5_slider.move(160, 370)
		self.layer5_slider.setEnabled(False)

		self.activtion_layer5 = QComboBox(self)
		self.activtion_layer5.setGeometry(350, 350, 80, 30)
		self.activtion_layer5.move(200, 370)
		self.activtion_layer5.setEnabled(False)
		self.activtion_layer5.addItems(['linear', 'sigmoid', 'relu', 'tanh'])

		self.train_model_btn = QPushButton('Train Model', self)
		self.train_model_btn.setGeometry(350, 350, 100, 30)
		self.train_model_btn.move(460, 410)
		self.train_model_btn.clicked.connect(self.trainModel)
		self.train_model_btn.setEnabled(False)

		self.export_code_btn = QPushButton('Export Code', self)
		self.export_code_btn.setGeometry(350, 350, 100, 30)
		self.export_code_btn.move(560, 410)
		self.export_code_btn.clicked.connect(self.export_code)
		self.export_code_btn.setEnabled(False)

		self.optimizer = QComboBox(self)
		self.optimizer.setGeometry(350, 350, 120, 30)
		self.optimizer.move(360, 310)
		self.optimizer.addItems(['Choose Optimizer', 'GradientDescent', 'Adagrad', 'Momentum', 'Adam', 'RMSProp'])
		self.optimizer.activated.connect(self.update_ui)
		self.optimizer.setEnabled(False)

		self.temp1_btn_text = QLabel(self)
		self.temp1_btn_text.setGeometry(350, 350, 90, 30)
		self.temp1_btn_text.move(310, 350)
		self.temp1_btn_text.setText('beta1')
		self.temp1_btn_text.setVisible(False)

		self.temp1_btn = QDoubleSpinBox(self)
		self.temp1_btn.setGeometry(350, 350, 100, 30)
		self.temp1_btn.move(300, 370)
		self.temp1_btn.setDecimals(5)
		self.temp1_btn.setRange(0.00000001, 1)
		self.temp1_btn.setEnabled(False)
		self.temp1_btn.setVisible(False)

		self.temp2_btn_text = QLabel(self)
		self.temp2_btn_text.setGeometry(350, 350, 70, 30)
		self.temp2_btn_text.move(430, 350)
		self.temp2_btn_text.setText('beta2')
		self.temp2_btn_text.setVisible(False)

		self.temp2_btn = QDoubleSpinBox(self)
		self.temp2_btn.setGeometry(350, 350, 90, 30)
		self.temp2_btn.move(420, 370)
		self.temp2_btn.setDecimals(5)
		self.temp2_btn.setRange(0.00000001, 1)
		self.temp2_btn.setEnabled(False)
		self.temp2_btn.setVisible(False)

		self.temp3_btn_text = QLabel(self)
		self.temp3_btn_text.setGeometry(350, 350, 50, 30)
		self.temp3_btn_text.move(490, 350)
		self.temp3_btn_text.setText('gamma')
		self.temp3_btn_text.setVisible(False)

		self.temp3_btn = QDoubleSpinBox(self)
		self.temp3_btn.setGeometry(350, 350, 75, 30)
		self.temp3_btn.move(480, 370)
		self.temp3_btn.setDecimals(5)
		self.temp3_btn.setRange(0.00000001, 1)
		self.temp3_btn.setEnabled(False)
		self.temp3_btn.setVisible(False)

		self.evaluate_btn = QPushButton('Evalauate Model', self)
		self.evaluate_btn.setGeometry(350, 350, 180, 30)
		self.evaluate_btn.move(330, 480)
		self.evaluate_btn.clicked.connect(self.evaluateModel)
		self.evaluate_btn.setEnabled(False)

		self.train_f1_score = QLineEdit(self)
		self.train_f1_score.setGeometry(350, 350, 180, 30)
		self.train_f1_score.move(330, 510)

		self.test_f1_score = QLineEdit(self)
		self.test_f1_score.setGeometry(350, 350, 180, 30)
		self.test_f1_score.move(330, 540)

		self.statusBar()

		openFile = QAction(QIcon('open.png'), 'Open', self)
		openFile.setShortcut('Ctrl+O')
		openFile.setStatusTip('Open new File')
		openFile.triggered.connect(self.showDialog)

		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		fileMenu.addAction(openFile)
		self.setGeometry(300, 300, 850, 600)
		self.setWindowTitle('Layers')
		self.show()

	def print_output(self, d):
		self.data = d

	def toggle1(self):
		print("State : Button is %s" %(str(self.l1_regularization.isChecked())))
		if self.l1_regularization.isChecked():
			# self.l1_regularization.setChecked(True)
			self.lambda1.setEnabled(True)
		else:
			# self.l1_regularization.setChecked(False)
			self.lambda1.setEnabled(False)

	def toggle2(self):
		print("State : Button is %s" %(str(self.l2_regularization.isChecked())))
		if self.l2_regularization.isChecked():
			# self.l1_regularization.setChecked(True)
			self.lambda2.setEnabled(True)
		else:
			# self.l1_regularization.setChecked(False)
			self.lambda2.setEnabled(False)

	def toggle3(self):
		print("State : Button is %s" %(str(self.dropout_btn.isChecked())))
		if self.dropout_btn.isChecked():
			# self.l1_regularization.setChecked(True)
			self.dropout.setEnabled(True)
		else:
			# self.l1_regularization.setChecked(False)
			self.dropout.setEnabled(False)


	def load_data(self, fname):

		if fname[0] is not '':
			print(fname[0])
			ext = fname[0].split('.')[-1]
			print("load_data")
			# w1 = Worker(self.load_subdata, fname)
			# self.threadpool.start(w1)

			if ext == 'tsv':

				data = pd.read_csv(fname[0], sep ='\s+',prefix='X',low_memory=False, header=None)
				print(data[:2])
			else:
				data = pd.read_csv(fname[0], low_memory=False)

			# w1.signals.result.connect(self.progress_show)
			# w1.signals.finished.connect(self.thread_complete_sub)

		return data

	def load_subdata(self, fname):

		if fname[0] is not '':
			print(fname[0])
			ext = fname[0].split('.')[-1]

			num_lines = int(subprocess.check_output(['wc', '-l', fname[0]]).split()[0])
			print("Called from load_subdata, num_lines = %d " %(num_lines))
			if ext == 'tsv':
				start = time.time()
				data = pd.read_csv(fname[0], sep ='\s+',prefix='X',low_memory=False, header=None, nrows=100)
				end = time.time()

				print(data[:2])
			else:
				start = time.time()
				data = pd.read_csv(fname[0], low_memory=False, nrows=100)
				end = time.time()

			elapsed_time = end - start

		return data

	def thread_complete(self):
		if self.counter > 0:
			self.counter -= 1
		self.train_model_btn.setEnabled(False)
		self.train_model_btn.setText('Train Model')
		self.train_model_btn.setStyleSheet("background-color: rgb(210, 221, 242)")

		self.layer1_slider.setEnabled(False)
		self.layer2_slider.setEnabled(False)
		self.layer3_slider.setEnabled(False)
		self.layer4_slider.setEnabled(False)
		self.layer5_slider.setEnabled(False)

		self.activtion_layer1.setEnabled(False)
		self.activtion_layer2.setEnabled(False)
		self.activtion_layer3.setEnabled(False)
		self.activtion_layer4.setEnabled(False)
		self.activtion_layer5.setEnabled(False)

		self.l1_regularization.setEnabled(False)
		self.l2_regularization.setEnabled(False)

		# self.dropout.setEnabled(True)
		self.dropout_btn.setEnabled(False)

		self.optimizer.setEnabled(False)
		# self.learning_rate.setEnabled(True)

		self.setStatusTip('!!! Data loaded sucessfully!!!"')

		# self.load_btn.setStyleSheet(" background-color: green")

		self.select_target_variable.clear()
		self.select_target_variable.addItems(list(self.data.columns.values[::-1]))
		self.select_target_variable.setEnabled(True)

	def thread_complete_sub(self):
		self.flag = 1
		print(self)
		# self.load_btn.setStyleSheet(" background-color: blue")
			# time.sleep(5)

		# self.load_btn.setStyleSheet(" background-color: green")

	def progress_fn(self, n):

		self.load_btn.setStyleSheet(" background-color: blue")
		self.setStatusTip("Data is being loaded. It will take %d seconds" % 10*n + 2)
		v = 0
		print("Val: %d" % n)
		self.progressbar.setValue(n)

	def progress_show(self, n):
		print("Val : %d" % n)
		v = 0
		while v < 100:
			self.progressbar.setValue(v)
			time.sleep(1)
			v += 10/float(n)
		v = 100
		self.progressbar.setValue(v)

	def showDialog(self):
		self.fname = QFileDialog.getOpenFileName(self, "Open File", '/home',
			"Tabular (*csv *.tsv)")
		# w1 = Worker(self.load_subdata, fname)

		w = Worker(self.load_subdata, self.fname)
		w.signals.result.connect(self.print_output)
		w.signals.finished.connect(self.thread_complete)
		# w.signals.progress.connect(self.progress_fn)

		# self.threadpool.start(w1)
		self.threadpool.start(w)


	def selectionchange(self, i):
		self.target_variable = self.select_target_variable.currentText()
		print(self.target_variable)
		# self.trainData_btn.setEnabled(False)
		# self.testData_btn.setEnabled(False)
		# self.trainData_btn.setStyleSheet("background-color: white")
		# self.testData_btn.setStyleSheet("background-color: white")

		if self.target_variable != 'Select Target Variable':
			self.split_btn.setEnabled(True)
			self.ratio.clear()
			self.ratio.addItems([str(i/10) for i in range(0,11)])
			self.ratio.setEnabled(True)

		else:
			self.split_btn.setEnabled(False)
		# 	def splitData(self):

		# 		if self.target_variable != 'Select Target Variable':
		# 			self.Y = self.data['%s' % self.target_variable]
		# 			self.X = self.data.drop('%s' % self.target_variable, axis=1)
		# 			self.X_train, self.X_test,
		# self.y_train, self.y_test =train_test_split(self.X,self.Y, test_size = 0.2)
		# 			self.trainData_btn.setEnabled(True)
		# 			self.testData_btn.setEnabled(True)
		# 			self.train_model_btn.setEnabled(True)

	def selectTrain(self):
		self.trainData_btn.setStyleSheet("background-color: rgb(210, 221, 242)")

	def selectTest(self):
		self.testData_btn.setStyleSheet("background-color: rgb(210, 221, 242)")
		# self.trainData_btn.setEnabled(False)

	def splitData(self, ratio=0.8):
		# Filtering out the variables that
		# have very less unique values using simple statistics.
		if self.target_variable != 'Select Target Variable':

			isCat = []
			drop_ = []
			count = 0
			for col in self.data.columns:
				if self.data[col].nunique() > 1:

					if 1. * self.data[col].nunique() / self.data[col].count() < 0.0001:
						isCat.append(col)
						uniq = self.data[col].unique()

						# print("\nThe number of unique\
						# 	values in Variable %s is %d" % (col, len(uniq)))
						print(uniq)
						count += len(uniq)

				else:
					drop_.append(col)

			# Drop columns that have only one unique value.
			filtered_data = self.data.drop(drop_, axis=1)

			# Split the training dataset as
			# 75,15,10 % respectively for train,validation and test.
			ratio = float(self.ratio.currentText())
			print("shape: %s" % str(self.data.shape))
			Y = filtered_data[self.target_variable]
			Y = Y.astype('str')
			X = filtered_data.drop([self.target_variable], axis=1)
			self.target_outputs = Y.unique()
			print("Target outputs: %s, len: %d" % (str(self.target_outputs), len(self.target_outputs)))

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

			self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_norm, Y, test_size = ratio, random_state = 145)
			lb = LabelBinarizer()
			# one_hot.fit(self.target_outputs)
			print("self.y_train: %s" %(self.target_outputs[:2]))
			lb.fit(self.target_outputs)
			self.y_train_ = lb.transform(self.y_train)
			# print("one-hot lambda2ables: %s" %(one_hot.lables))
			print("lb classes : %s" % str(lb.classes_))
			print("y_train_: %s " % str(self.y_train_[:2]))
			self.y_test_ = lb.transform(self.y_test)

			if len(self.target_outputs) == 2:
				self.y_train_ = np.hstack((1 - self.y_train_, self.y_train_))
				self.y_test_ = np.hstack((1 - self.y_test_, self.y_test_))
			# self.y_val_ = lb.transform(self.y_val)

			self.y_train_prob = pd.np.array(self.y_train_, dtype=pd.np.float64)
			self.y_test_prob = pd.np.array(self.y_test_, dtype=pd.np.float64)
			print("y_train_prob: %s" %str(self.y_train_prob[:2]))
			# self.y_val_prob = pd.np.array(self.y_val_, dtype=pd.np.float64)

			# self.trainData_btn.setEnabled(True)
			# self.testData_btn.setEnabled(True)
			self.train_model_btn.setEnabled(True)

			self.layer1_slider.setEnabled(True)
			self.layer2_slider.setEnabled(True)
			self.layer3_slider.setEnabled(True)
			self.layer4_slider.setEnabled(True)
			self.layer5_slider.setEnabled(True)

			self.layer1_slider.setMinimum(1)
			self.layer1_slider.setMaximum(1000)
			self.layer2_slider.setMinimum(1)
			self.layer2_slider.setMaximum(1000)
			self.layer3_slider.setMinimum(1)
			self.layer3_slider.setMaximum(1000)
			self.layer4_slider.setMinimum(1)
			self.layer4_slider.setMaximum(1000)
			self.layer5_slider.setMinimum(1)
			self.layer5_slider.setMaximum(1000)

			self.activtion_layer1.setEnabled(True)
			self.activtion_layer2.setEnabled(True)
			self.activtion_layer3.setEnabled(True)
			self.activtion_layer4.setEnabled(True)
			self.activtion_layer5.setEnabled(True)

			self.l1_regularization.setEnabled(True)
			self.l2_regularization.setEnabled(True)

			# self.dropout.setEnabled(True)
			self.dropout_btn.setEnabled(True)

			self.optimizer.setEnabled(True)
			self.export_code_btn.setEnabled(True)
			# self.learning_rate.setEnabled(True)

			self.setStatusTip('!!! Data splitted sucessfully!! Choose number of neurons for each layer')
			print(self.X_train[:5])

	def print_actfn(self):
			print(self.actv_fn(self.activtion_layer1.currentText()))

	def actv_fn(self, name):
		if name == ('tanh' or 'sigmoid'):
			return getattr(tf, name)
		else:
			return getattr(tf.nn, name)

	def actv_fn_str(self, name):
		if name == ('tanh' or 'sigmoid'):
			return 'tf.' + name
		else:
			return 'tf.nn.' + name

	def update_ui(self):

		if self.optimizer.currentText() != 'Choose Optimizer':
			func = getattr(tf.train, self.optimizer.currentText() + 'Optimizer')
			args = []
			sig = inspect.signature(func)

			dic = sig.parameters

			for key in dic.items():
				val = key[1].default

				if type(val).__name__ != 'type':
					continue
				else:
					args.append(key[0])

			print("Args: %s" % str(args))
			if len(args) > 0:
				for i in range(len(args)):
					print(i, args[i])
					text_function = getattr(self, 'temp%d_btn_text' % (i + 1))
					text_function.setVisible(True)
					text_function.setText(args[i])
					btn = getattr(self, 'temp%d_btn' % (i + 1))
					btn.setVisible(True)
					btn.setEnabled(True)


				for j in range(i + 1, 3):
					btn = getattr(self, 'temp%d_btn' % (j + 1))
					btn.setVisible(False)
					btn.setEnabled(False)

					text_function = getattr(self, 'temp%d_btn_text' % (j + 1))
					text_function.setVisible(False)

			else:
				for i in range(3):
					btn = getattr(self, 'temp%d_btn' % (i + 1))
					btn.setVisible(False)
					btn.setEnabled(False)

					text_function = getattr(self, 'temp%d_btn_text' % (i + 1))
					text_function.setVisible(False)

			self.optimizer_func, self.args = func, args


	def build_dict(self):
		di = {}

		for i in range(5):
			func = getattr(self, 'activtion_layer%d' % (i + 1))
			di['activation_layer%d' % (i+1)] = '"%s"' % self.actv_fn_str(func.currentText())

		for i in range(len(self.args)):
			btn = getattr(self, 'temp%d_btn' % (i + 1))
			di[self.args[i]] = btn.value()

		di['optimizer_func'] = 'tf.train.' + self.optimizer_func.__name__ + '('
		for i in range(len(self.args)):
			val_func = getattr(self, 'temp%d_btn' % (i + 1))
			di['optimizer_func'] += '%s=%f,' %(self.args[i],float(val_func.value()))

		di['optimizer_func'] += ')'
		di['ratio'] = float(self.ratio.currentText())


		if self.l1_regularization.isChecked():
			di['l1_regularization'] = self.lambda1.value()
		else:
			di['l1_regularization'] = '""'

		if self.l2_regularization.isChecked():
			di['l2_regularization'] = self.lambda2.value()
		else:
			di['l2_regularization'] = '""'

		if self.dropout_btn.isChecked():
			di['dropout'] = self.dropout.value()
			di['dropout_btn'] = True
		else:
			di['dropout_btn'] = False
		hidden_layers = [self.layer1_slider.value(), self.layer2_slider.value(), self.layer3_slider.value(), self.layer4_slider.value(), self.layer5_slider.value()]

		di['hidden_layers'] = hidden_layers
		return di

	def export_code(self):
		params = self.build_dict()


		f = open('code.py', 'w')

		temp = open('imports.py', 'r')

		for l in temp.readlines():
			f.write('%s' % l)

		f.write('\n')

		for key, val in params.items():
			f.write('%s = %s\n' % (key, str(val)))
		temp.close()

		f.write('file_name = "%s"\n' % self.fname[0])


		f.write('\n\n# Function for loading data using pandas read_csv or tsv\n\n')

		temp = open('load_data.py', 'r')
		for l in temp.readlines():
			f.write('%s' % l)

		f.write('\n\ndata = load_data(file_name)\n' )
		f.write('\ntarget_variable = "%s"\n\n' % self.target_variable)
		temp.close()

		temp = open('split_data.py', 'r')
		for l in temp.readlines():
			f.write('%s' % l)
		temp.close()

		f.write('\nX_train, X_test, y_train_prob, y_test_prob, target_outputs = splitData(ratio)\n\n')
		# f.write('\nprint(X_train[:2])\n')

		temp = open('train_model.py', 'r')
		for l in temp.readlines():
			f.write('%s' % l)
		temp.close()

		f.write('\n\nprint(trainModel())')
		f.close()


	def trainModel(self):

		# Todo
		if self.counter < 1:
			self.data = self.load_data(self.fname)
			self.splitData()

		x = tf.placeholder(tf.float64, shape=[None, self.X_train.shape[1]], name='x')
		print("Len of target outputs: %d " % len(self.target_outputs))
		y_true = tf.placeholder(tf.float64, shape=[None, len(self.target_outputs)], name='y_true')

		# Let us build a simple two layer neural network 

		hidden_layers = [self.X_train.shape[1], self.layer1_slider.value(), self.layer2_slider.value(), self.layer3_slider.value(), self.layer4_slider.value(), self.layer5_slider.value(), len(self.target_outputs)]
		# self.layers = hidden_layers
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
		if self.activtion_layer1.currentText() != 'linear':
			act1 = self.actv_fn(self.activtion_layer1.currentText())
			out1 = act1(out1)
		# out1 = tf.sigmoid(out1)
		if self.dropout_btn.isChecked():
			out1 = tf.nn.dropout(out1, self.dropout.value())

		out2 = tf.matmul(out1, W2) + b2
		if self.activtion_layer2.currentText() != 'linear':
			act2 = self.actv_fn(self.activtion_layer2.currentText())
			out2 = act2(out2)
		# out2 = tf.sigmoid(out2)
		if self.dropout_btn.isChecked():
			out2 = tf.nn.dropout(out2, self.dropout.value())

		out3 = tf.matmul(out2, W3) + b3
		if self.activtion_layer3.currentText() != 'linear':
			act3 = self.actv_fn(self.activtion_layer3.currentText())
			out3 = act3(out3)
		if self.dropout_btn.isChecked():
			out3 = tf.nn.dropout(out3, self.dropout.value())

		out4 = tf.matmul(out3, W4) + b4
		if self.activtion_layer4.currentText() != 'linear':
			act4 = self.actv_fn(self.activtion_layer4.currentText())
			out4 = act4(out4)
		if self.dropout_btn.isChecked():
			out4 = tf.nn.dropout(out4, self.dropout.value())

		out5 = tf.matmul(out4, W5) + b5
		if self.activtion_layer5.currentText() != 'linear':
			act5 = self.actv_fn(self.activtion_layer5.currentText())
			out5 = act5(out5)
		print(self.dropout_btn.isChecked())
		if self.dropout_btn.isChecked():
			out5 = tf.nn.dropout(out5, self.dropout.value())

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

		l1_reg = tf.contrib.layers.l1_regularizer(scale=self.lambda2.value(), scope=None)


		loss1 = tf.nn.weighted_cross_entropy_with_logits(
			weighted_logits, y_true, pos_weight = 0.99, name = 'loss1')
		reg = 0
		if self.l1_regularization.isChecked():
			print("l1 activated")
			reg += self.lambda1.value()*(regularizer6 + regularizer5 + regularizer4 + regularizer3 + regularizer2 + regularizer1)
		if self.l2_regularization.isChecked():
			print("L2 activated")
			reg += tf.contrib.layers.apply_regularization(l1_reg, weights)
		cost = tf.reduce_mean(loss + reg)




		val_args = []
		for i in range(len(self.args)):
			val_func = getattr(self, 'temp%d_btn' %(i+1))
			val_args.append(float(val_func.value()))

		if self.optimizer.currentText() == 'Choose Optimizer':
			return 
		optimizer = self.optimizer_func(*val_args).minimize(cost)

		sess = tf.Session()
		init = tf.initialize_all_variables()

		sess.run(init)
		feed_dict_train = {x: self.X_train.values, y_true: self.y_train_prob}
		feed_dict_test = {x: self.X_test.values, y_true: self.y_test_prob}
		# feed_dict_val = {x:self.X_val.values,y_true: self.y_val_prob}

		from sklearn.metrics import confusion_matrix,f1_score,precision_recall_fscore_support,roc_curve,roc_auc_score
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
		self.f1_sco_test = test_results

		cof,y_true_,y_pred_,y_pred_prob = sess.run([tf.confusion_matrix(y_true_cls,y_pred_cls),y_true_cls,y_pred_cls,y_pred],feed_dict_train)
		train_results = accuracy_score(y_true_,y_pred_)

		self.f1_sco_train = train_results
		self.evaluate_btn.setEnabled(True)
		self.counter += 1
		self.train_model_btn.setStyleSheet("background-color: pink")
		self.train_model_btn.setText("Retrain Model")

	def evaluateModel(self):
		self.test_f1_score.setText('Test Accuracy: %f' %(self.f1_sco_test))
		self.train_f1_score.setText('Train Accuracy: %f' %(self.f1_sco_train))



if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = Example()
	sys.exit(app.exec_())
