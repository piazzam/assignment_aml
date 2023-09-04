import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import Input
from keras.models import Model
from tensorflow.keras import regularizers
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint


def display_figure(figure):
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(figure, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

def plot_history_nn(network_history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(x_plot, network_history.history['loss'])
    plt.plot(x_plot, network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(x_plot, network_history.history['accuracy'])
    plt.plot(x_plot, network_history.history['val_accuracy'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()

def plot_history_ae(network_history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(x_plot, network_history.history['loss'])
    plt.plot(x_plot, network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.plot(x_plot, network_history.history['mse'])
    plt.plot(x_plot, network_history.history['val_mse'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()

x_train = pickle.load(open("dataset/x_train.obj","rb"))

#x_test = pickle.load(open("dataset/x_test.obj","rb"))

y_train = pickle.load(open("dataset/y_train.obj","rb"))

y_train = [y - 16 for y in y_train]
y_train = np.array(y_train)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train)
yy_val = y_val

y_train = np_utils.to_categorical(y_train, 11)
y_val = np_utils.to_categorical(y_val, 11)
print(np.unique(y_train, return_counts = False))

display_figure(x_train[0])

x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_val = x_val.reshape((len(x_val), np.prod(x_val.shape[1:])))

print(x_train.shape)

nb_classes = 11

print (x_train.shape, y_train.shape)

model = Sequential()

initializer = tf.keras.initializers.GlorotUniform(seed=1234)

model.add(Dense(256, input_shape=(784,), activation = "relu",kernel_initializer=initializer))
model.add(Dropout(0.3))
model.add(Dense(128, activation = "relu", kernel_initializer=initializer))
model.add(Dropout(0.2))
model.add(Dense(32, activation = "relu", kernel_initializer=initializer))
model.add(Dropout(0.1))
model.add(Dense(16, activation = "relu", kernel_initializer=initializer))
model.add(Dropout(0.05))
model.add(Dense(nb_classes, activation = "softmax", kernel_initializer=initializer))

from tensorflow.keras.optimizers import SGD
model.compile(optimizer = "adam",
              loss = "categorical_crossentropy",
              metrics = ["accuracy"]
              )

model.summary()

n_epochs = 100
batch_size = 128

fBestModel = 'best_model.h5'
early_stop = EarlyStopping(monitor='val_loss', patience=8, min_delta = 0.1, verbose=1, restore_best_weights=True)
best_model = ModelCheckpoint(fBestModel, verbose=0, save_best_only=True)

history = model.fit(x_train, y_train, epochs = n_epochs, batch_size = batch_size,
                    verbose=2, validation_data=(x_val, y_val), callbacks=[best_model, early_stop])


x_plot = list(range(1,len(history.history["loss"])+1))
plot_history_nn(history)

predictions = model.predict(x_val) 
print('predictions shape:', predictions.shape)

y_classes = predictions.argmax(axis=-1)

from sklearn.metrics import classification_report
print(classification_report(y_classes, yy_val))

from sklearn.metrics import precision_score
precision = precision_score(y_classes, yy_val, average = 'macro')
print(precision)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_classes, yy_val)
print(accuracy)

from sklearn.metrics import recall_score
recall = recall_score(y_classes, yy_val, average = 'weighted')
print(recall)

from sklearn.metrics import f1_score
f1 = f1_score(y_classes, yy_val, average = 'weighted')
print(f1)

x_train = pickle.load(open("dataset/x_train.obj","rb"))
display_figure(x_train[0])

x_train = x_train.astype('float32') / 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

x_train, x_val = train_test_split(x_train)


print(x_train.shape)
print(x_val.shape)

encoding_dim = 64

input_img = Input(shape=(784,))

encoded = Dense(encoding_dim, activation='relu')(input_img)

decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer= "adam", loss='binary_crossentropy', metrics = ["mse"])
autoencoder.summary()
history = autoencoder.fit(x_train,x_train,
                epochs=50,
                batch_size=512,
                shuffle=True,
                validation_data=(x_val, x_val))

x_plot = list(range(1,len(history.history["loss"])+1))
plot_history_ae(history)

x_selected = x_val

decoded_imgs = autoencoder.predict(x_selected)

n = 10
plt.figure(figsize=(20, 4))

for i in range(n):

    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_selected[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

x_train = pickle.load(open("dataset/x_train.obj","rb"))
y_train = pickle.load(open("dataset/y_train.obj","rb"))

y_train = [y - 16 for y in y_train]

display_figure(x_train[0])

x_train = x_train.astype('float32') / 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

x_train_ae, x_val_ae = train_test_split(x_train)

print(x_train_ae.shape)
print(x_train_ae.shape)

encoding_dim = 32

input_img = Input(shape=(784,))

encoded = Dense(encoding_dim, activation='relu')(input_img)

decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer= "adam", loss='binary_crossentropy', metrics = ["mse"])
autoencoder.summary()
history = autoencoder.fit(x_train_ae,x_train_ae,
                epochs=50,
                batch_size=512,
                shuffle=True,
                validation_data=(x_val_ae, x_val_ae))

x_plot = list(range(1,len(history.history["loss"])+1))
plot_history_ae(history)

encoder = Model(input_img, encoded)
encoded_data = encoder.predict(x_train)
    
    
x_train_bis, x_val, y_train_bis, y_val = train_test_split(encoded_data, y_train)


from sklearn import svm

model_svm = svm.SVC(C = 10)
    
model_svm.fit(x_train_bis, y_train_bis)

y_classes = model_svm.predict(x_val)

from sklearn.metrics import classification_report
print(classification_report(y_classes, y_val))

from sklearn.metrics import precision_score
precision = precision_score(y_classes, y_val, average = 'macro')
print(precision)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_classes, y_val)
print(accuracy)
    
from sklearn.metrics import recall_score
recall = recall_score(y_classes, y_val, average = 'weighted')
print(recall)


from sklearn.metrics import f1_score
f1 = f1_score(y_classes, y_val, average = 'weighted')
print(f1)

x_test = pickle.load(open("dataset/x_test.obj","rb"))

x_test = x_test.astype('float32') / 255
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

encoded_x_test = encoder.predict(x_test)

y_predicted = model_svm.predict(encoded_x_test)

np.savetxt("result.txt", y_predicted,fmt='%d')
