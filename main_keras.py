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
import argparse
import sys
import os 

DATA = "/home/ubuntu/Desktop/lab_project/diabetes.csv"
CHECK = '/home/ubuntu/Desktop/lab_project/keras/model'
CHECKh5='/home/ubuntu/Desktop/lab_project/keras/modelh5'
PLOT = 'no'
def parse_arguments(argv):
	parser = argparse. ArgumentParser()

	parser.add_argument('--dataset', type = str,default=DATA, help='Directory to dataset')
	parser.add_argument('--checkckpt_dir',type = str,default=CHECK, help='checkpont logs')
	parser.add_argument('--checkh5_dir',type = str,default=CHECKh5, help='checkpont logs')
	parser.add_argument('--plot',type = str,default=PLOT, help='yes/no')

	return parser.parse_args(argv)
def plot_history(history):
	print(history.history.keys())
# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

def create_model():
	model = Sequential()
	model.add(Dense(12, input_dim=11, kernel_initializer='uniform', activation='relu'))
	model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
	model.add(Dense(4, kernel_initializer='uniform', activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(2, kernel_initializer='uniform', activation='softmax'))
	# Compile model
	model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy', metrics = ['accuracy'])
	return model
def main(args):
	dataset = args.dataset
	check_dir = args.checkckpt_dir
	checkh5_dir=args.checkh5_dir

	if not os.path.isdir(check_dir):
		os.makedirs(check_dir)
	else:
		pass

	if not os.path.isdir(checkh5_dir):
		os.makedirs(checkh5_dir)
	else:
		pass

	seed = 7
	numpy.random.seed(seed)
	# load pima indians dataset
	dataset = numpy.loadtxt(dataset, delimiter=",", skiprows=1)
	# split into input (X) and output (Y) variables
	X = dataset[:,0:8]
	Y = dataset[:,8]
	# create model

	# Fit the model

	(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.1, random_state=seed)
	model = create_model()
	# model.fit(X, Y, validation_data=(X_test, Y_test), epochs=100, batch_size=10, verbose=1)
	history = model.fit(X, Y, validation_data=(X_test, Y_test), epochs=500, batch_size=10, verbose=1)
	# list all data in history

	model_json = model.to_json()
	with open("%s/model.json"%checkh5_dir, "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("%s/model.h5"%checkh5_dir)

	saver = tf.train.Saver()
	sess = keras.backend.get_session()
	save_path = saver.save(sess, "%s/model.ckpt"%check_dir)
	print("Saved model to disk")


	scores = model.evaluate(X_test, Y_test)
	print("Accuracy: %.2f%%" % (scores[1]*100))

	if args.plot=='yes':
		plot_history(history)
	else:
		print ('Thanks for using Quang`s Product')


if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
