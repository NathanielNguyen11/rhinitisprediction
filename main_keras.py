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

DATA = '.\dataset_new_label_ANN.csv'
CHECK = '.\model'
CHECKh5='.\modelh5'
PLOT = 'no'
def parse_arguments(argv):
	parser = argparse. ArgumentParser()

	parser.add_argument('--dataset', type = str,default=DATA, help='Directory to dataset')
	parser.add_argument('--checkckpt_dir',type = str,default=CHECK, help='checkpoint logs')
	parser.add_argument('--checkh5_dir',type = str,default=CHECKh5, help='checkpoint logs')
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
	# model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
	model.add(Dense(4, kernel_initializer='uniform', activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(2, kernel_initializer='uniform', activation='softmax'))
	# Compile model
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
def main(args):
	dataset = args.dataset
	check_dir = args.checkckpt_dir
	checkh5_dir = args.checkh5_dir

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
	X = dataset[:,0:11]
	Y = dataset[:,11]
	scaler = MinMaxScaler(feature_range=(0, 1))
	rescaledX = scaler.fit_transform(X)
	print(X)
	print(Y.shape)
	# encoder = LabelEncoder()
	# encoder.fit(Y)
	# encoded_Y = encoder.transform(Y)

	(X_train, X_test, Y_train, Y_test) = train_test_split(rescaledX, Y, test_size=0.1, random_state=seed)
	print(Y_train)
	model = create_model()
	# model.fit(X, Y, validation_data=(X_test, Y_test), epochs=100, batch_size=10, verbose=1)
	history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=10, verbose=1)
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
	#
	#
	scores = model.evaluate(X_test, Y_test)
	Y_pro = model.predict(X_test)
	y_pred = model.predict_classes(X_test)
	print (y_pred)
	print (Y_pro)
	
	print("Accuracy: %.2f%%" % (scores[1]*100))

	if args.plot=='yes':
		plot_history(history)
	else:
		print ('Thanks for using Quang`s Product')


if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))
