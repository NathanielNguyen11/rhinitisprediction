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


DATA = '/home/ubuntu/Desktop/lab_project/diabetes_test.csv'
CHECK_DIR = '/home/ubuntu/Desktop/lab_project/keras/modelh5'

def parse_arguments(argv):
	parser = argparse. ArgumentParser()
	parser.add_argument('--dataset', type = str,default=DATA, help='Directory to dataset')
	parser.add_argument('--check_dir',type = str,default=CHECK_DIR, help='checkpont logs')

	return parser.parse_args(argv)

# fix random seed for reproducibility
def main(args):
	seed = 7
	np.random.seed(seed)
	# load pima indians dataset
	dataset = np.loadtxt("/home/ubuntu/Desktop/lab_project/diabetes.csv", delimiter=",", skiprows=1)
	# split into input (X) and output (Y) variables
	X = dataset[:,0:8]
	Y = dataset[:,8]

	model = create_model()

	(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=seed)

	# continue train
	filepath = "/home/ubuntu/Desktop/lab_project/keras/model.h5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	# fit the model
	model.fit(X_train, Y_train, epochs=5, batch_size=50, callbacks=callbacks_list)

	# load the model
	new_model = load_model(filepath)
	assert_allclose(model.predict(X_train),
	                new_model.predict(X_train),
	                1e-5)

	# fit the model
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	new_model.fit(X_train, Y_train, epochs=5, batch_size=50, callbacks=callbacks_list)
if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
