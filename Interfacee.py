#hide console at first
import win32gui, win32con

The_program_to_hide = win32gui.GetForegroundWindow()
win32gui.ShowWindow(The_program_to_hide , win32con.SW_HIDE)

from tkinter import *
from PIL import ImageTk, Image
import os
from tkinter import filedialog
import numpy
from main_keras import create_model
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from test import load_checkpoint
from main_keras import main

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
    (X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.1, random_state=seed)
    model = create_model()
    history = model.fit(X, Y, validation_data=(X_test, Y_test), epochs=200, batch_size=200, verbose=1)
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
    print(("Accuracy: %.2f%%" % (scores[1]*100)))
    a = training_accuracy.config(text=scores[1]*100)
    return a
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
    X_test = dataset[:,0:8]
    Y_test = dataset[:,8]

    loaded_model = load_checkpoint(checkh5_dir)
    loss, acc = loaded_model.evaluate(X_test, Y_test)
    Y_pro = loaded_model.predict(X_test)
    Y_predict = loaded_model.predict_classes(X_test)

    print(Y_predict.shape)
    for index in range(X_test.shape[0]):
        if Y_pro[index]*100 <= 50 and Y_predict[index]==0:
            Y_pro[index] = 100- Y_pro[index]*100
            # print ('Prediction class {}  with confidentscre :{:2.2f}%'.format(Y_predict[index],float(Y_pro[index]*100)))
        else:
            Y_pro[index] = Y_pro[index]*100
        print(('Prediction class {}  with confidence :{:5.2f}%  label {}'.
            format(Y_predict[index],float(Y_pro[index]),Y_test[index])))

    print(("Untrained model, accuracy: {:5.2f}%".format(100*acc)))
    print('Success in prediction ')

    b = t_accuracy.config(text=100*acc)
    return b
def browse_original_training():
    global folder_path
    # filename = filedialog.askdirectory()
    root.filename_train_original = filedialog.askopenfilename()
    folder_path.set(root.filename_train_original)
    train_original_dataset = root.filename_train_original
    scores = main(train_original_dataset)
    o_training_accuracy.config(text=scores[1]*100)



if __name__ == '__main__':

    root = Tk()
    folder_path = StringVar()
    lbl1 = Label(master=root, textvariable=folder_path)
    root.title('Pattern Recognition and Machine Learning Lab')
    root.geometry('{}x{}'.format(760, 450))

    # create all of the main containers
    top_frame = Frame(root, bg='cyan', width=750, height=70, pady=3)
    center = Frame(root, bg='gray2', width=50, height=40, padx=3, pady=3)
    btm_frame = Frame(root, bg='white', width=450, height=45, pady=3)


    # layout all of the main containers
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)

    top_frame.grid(row=0, sticky="ew")
    center.grid(row=1, sticky="nsew")
    btm_frame.grid(row=3, sticky="ew")


    # create the widgets for the top frame
    model_label = Label(top_frame, text='Welcome to Korhina Simulator')
    width_label = Label(top_frame, text='-PRML lab')
    # length_label = Label(top_frame, text='Length:')
    # entry_W = Entry(top_frame, background="pink")
    # entry_L = Entry(top_frame, background="orange")

    # layout the widgets in the top frame
    model_label.grid(row=0, column=7)
    width_label.grid(row=0, column=9)
    # length_label.grid(row=1, column=2)
    # entry_W.grid(row=1, column=1)
    # entry_L.grid(row=1, column=3)

    # create the center widgets
    center.grid_rowconfigure(0, weight=1)
    center.grid_columnconfigure(1, weight=1)

    ctr_left = Frame(center, bg='lavender', width=400, height=190)
    ctr_mid = Frame(center, bg='lavender', width=350, height=190, padx=3, pady=3)


    ctr_left.grid(row=0, column=0, sticky="ns")
    ctr_mid.grid(row=0, column=1, sticky="nsew")


    C = Canvas(ctr_left, bg="blue")
    img_ori = Image.open("quang_formal.jpg")
    # print img_ori.size
    image = img_ori.resize((int(img_ori.size[0]*0.5),int(img_ori.size[1]*0.5)),Image.ANTIALIAS)
    img = ImageTk.PhotoImage(image)
    background_label = Label(ctr_left,image=img)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    background_label1 = Label(ctr_left,text='author',fg='blue')
    background_label1.place(x=20, y=20, relwidth=1, relheight=0)



    b = Button(ctr_mid, text="Select the csv file for training dataset", command=browse_training_button).place(x = 25,y = 5)
    e = Button(ctr_mid, text="Select the csv file for testing dataset", command=browse_testing_button).place(x=25, y=40)
    # Label(ctr_mid, text="Input number of features:").place(x=25, y=70)
    # Label(ctr_mid, text="Input number of features:").pack(side=LEFT, padx=5, pady=5)
    # User_input = Entry(ctr_mid)
    # User_input.pack()
    # entry = Entry(root)


    gg = Label(ctr_mid, text='Training Accuracy is:', fg='red')
    gg.place(x=25, y=85)
    training_accuracy = Label(ctr_mid,text='(Model not loaded)',fg='blue')
    training_accuracy.place(x=200,y=85)

    hh = Label(ctr_mid, text='Testing Accuracy is:', fg='red')
    hh.place(x=25, y=125)
    t_accuracy = Label(ctr_mid, text='(Model not loaded)', fg='blue')
    t_accuracy.place(x=200, y=125)



    add = Label(btm_frame,text='Pattern Recognition and Machine Learning Laboratory - Department of Software,Gachon University')
    add.place(x = 5,y=5)

    exit = Button(ctr_mid,text='Exit',command = root.destroy)
    exit.place(x = 75,y = 350)

    # C.pack()
    root.bind('<Return>')
    root.resizable(False,False)
    root.mainloop()
