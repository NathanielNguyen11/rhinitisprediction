import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import tensorflow as tf
from keras.layers import Dropout
# fix random seed for reproducibility
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import argparse
from keras.optimizers import SGD
import sys
import os


# checkpoint_path = "/home/ubuntu/Desktop/lab_project/keras/model/model.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)


TDATA = '.\dataset_new_label_ANN_test.csv'
CHECK_DIR = '.\modelh5'

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
	loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return loaded_model

def main(args):
	test_dataset = args.testdata
	check_dir = args.check_dir

	# load pima indians dataset
	dataset = numpy.loadtxt(test_dataset, delimiter=",", skiprows=1)
	# split into input (X) and output (Y) variables
	X_test = dataset[:,0:11]
	Y_test = dataset[:,11]
	scaler = MinMaxScaler(feature_range=(0, 1))
	rescaledX = scaler.fit_transform(X_test)
	# (X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=seed)
	# load model
	loaded_model = load_checkpoint(check_dir)

	loss, acc = loaded_model.evaluate(rescaledX, Y_test)
	Y_pro = loaded_model.predict(rescaledX)
	Y_predict = loaded_model.predict_classes(rescaledX)

	print(Y_pro)
	print(Y_predict)

	print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
	print('Success in prediction ')
if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
