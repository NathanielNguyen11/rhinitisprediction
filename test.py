from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import tensorflow as tf
import os
from main_keras import create_model
import argparse
import sys


# checkpoint_path = "/home/ubuntu/Desktop/lab_project/keras/model/model.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)


TDATA = '/home/ubuntu/Desktop/lab_project/diabetes_test.csv'
CHECK_DIR = '/home/ubuntu/Desktop/lab_project/keras/modelh5'

def parse_arguments(argv):
	parser = argparse. ArgumentParser()
	parser.add_argument('--testdata', type = str,default=TDATA, help='Directory to dataset')
	parser.add_argument('--check_dir',type = str,default=CHECK_DIR, help='checkpont logs')

	return parser.parse_args(argv)
def load_checkpoint (check_dir):

	json_file = open('%s/model.json'%check_dir, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("%s/model.h5"%check_dir)
	print("Loaded model from disk")
	loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	return loaded_model

def main(args):
	test_dataset = args.testdata
	check_dir = args.check_dir

	# load pima indians dataset
	dataset = numpy.loadtxt(test_dataset, delimiter=",", skiprows=1)
	# split into input (X) and output (Y) variables
	X_test = dataset[:,0:8]
	Y_test = dataset[:,8]
	# (X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=seed)
	# load model
	loaded_model = load_checkpoint(check_dir)

	loss, acc = loaded_model.evaluate(X_test, Y_test)
	Y_pro = loaded_model.predict(X_test)
	Y_predict = loaded_model.predict_classes(X_test)

	print (Y_predict.shape)
	for index in range(X_test.shape[0]):
		if Y_pro[index]*100 <= 50 and Y_predict[index]==0:
			Y_pro[index] = 100- Y_pro[index]*100
			# print ('Prediction class {}  with confidentscre :{:2.2f}%'.format(Y_predict[index],float(Y_pro[index]*100)))
		else:
			Y_pro[index] = Y_pro[index]*100
		print ('Prediction class {}  with confidentscre :{:5.2f}%  label {}'.
			format(Y_predict[index],float(Y_pro[index]),Y_test[index]))

	print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
	print('Success in prediction ')
if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
