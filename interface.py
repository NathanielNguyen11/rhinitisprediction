#hide console at first
import win32gui, win32con

The_program_to_hide = win32gui.GetForegroundWindow()
win32gui.ShowWindow(The_program_to_hide , win32con.SW_HIDE)


from tkinter import filedialog
from tkinter import *
import Tkconstants, tkFileDialog
from main_keras import create_model
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
import tensorflow as tf
# fix random seed for reproducibility
from test import load_checkpoint
import argparse
import sys
import os 

        
def browse_training_button():
    # Allow user to select a directory and store it in global var
    # called folder_path
    global folder_path
    # filename = filedialog.askdirectory()
    root.filename_train = filedialog.askopenfilename()
    folder_path.set(root.filename_train)
    # print
    # print(filename_train)
    # return root.filename_train
    dataset = root.filename_train
    check_dir = '%s/model'%(os.path.dirname(root.filename_train))
    checkh5_dir='%s/h5model'%(os.path.dirname(root.filename_train))
    if not os.path.isdir(check_dir):
        os.makedirs(check_dir,0777)
    else:
        pass

    if not os.path.isdir(checkh5_dir):
        os.makedirs(checkh5_dir,0777)
    else:
        pass
    seed = 7
    numpy.random.seed(seed)
    # load pima indians dataset
    dataset = numpy.loadtxt(dataset, delimiter=",", skiprows=1)
    # split into input (X) and output (Y) variables
    X = dataset[:,0:11]
    Y = dataset[:,11]
    (X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.1, random_state=seed)
    model = create_model()
    history = model.fit(X, Y, validation_data=(X_test, Y_test), epochs=2000, batch_size=100, verbose=1)
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
def browse_testing_button():
    global folder_path
    # filename = filedialog.askdirectory()
    root.filename_test = filedialog.askopenfilename()
    folder_path.set(root.filename_test)


    test_dataset = root.filename_test

    # dataset = root.filename_train
    check_dir = '%s/model'%(os.path.dirname(test_dataset))
    checkh5_dir='%s/h5model'%(os.path.dirname(test_dataset))
    dataset = numpy.loadtxt(test_dataset, delimiter=",", skiprows=1)
    X_test = dataset[:,0:11]
    Y_test = dataset[:,11]

    loaded_model = load_checkpoint(checkh5_dir)
    loss, acc = loaded_model.evaluate(X_test, Y_test)
    Y_pro = loaded_model.predict(X_test)
    Y_predict = loaded_model.predict_classes(X_test)

    print Y_predict.shape
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
    

    root = Tk()
    folder_path = StringVar()
    lbl1 = Label(master=root,textvariable=folder_path)
    root.geometry("500x500")
    root.resizable(0, 0)
    root.title('The inference')


    lbl1.grid(row=0, column=1)
    root.button1 = Button(text="Browse_training", command=browse_training_button)
    print root.button1
    root.button1.grid(row=0, column=0)
    button2 = Button(text="Browse_testing", command=browse_testing_button)
    button2.grid(row=1, column=0)
    root.bind('<Return>')

    mainloop()
