import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from keras.backend import abs, sum
import tensorflow as tf
from keras.layers import Dropout

def plot_history(network_history):
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


X_train = pd.read_csv('dataset/X_train.csv')
X_test = pd.read_csv("dataset/X_test.csv")
Y_train = pd.read_csv("dataset/y_train.csv")
df = pd.merge(X_train, Y_train, 
                   on='ID', 
                   how='inner')


df['MARRIAGE'] = np.where((df.MARRIAGE == 0), 3,df.MARRIAGE)
df['EDUCATION'] = np.where(((df.EDUCATION == 0) | (df.EDUCATION == 5) | (df.EDUCATION == 6)), 4, df.EDUCATION)
X_train = df.loc[:, df.columns!= 'default.payment.next.month']
Y_train = df['default.payment.next.month']

X_train = X_train.astype('float32')
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train)

nb_classes = 1
model = Sequential()

model.add(Dense(512, activation= "relu", input_shape=(24,)))
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(32, activation = "relu"))
model.add(Dropout(0.1))
model.add(Dense(1, activation = "sigmoid"))



from tensorflow.keras.optimizers import SGD
model.compile(optimizer = SGD(0.01),
              loss = "binary_crossentropy",
              metrics = ["accuracy"])

model.summary()

n_epochs = 50
batch_size = 128


history = model.fit(X_train, Y_train, epochs = n_epochs, batch_size = batch_size,
                    verbose=2, validation_data=(X_val, Y_val), class_weight = {1:0.65, 0:0.35})


x_plot = list(range(1,n_epochs+1))
plot_history(history)

predictions = model.predict(X_val) 
print('predictions shape:', predictions.shape)

from sklearn.metrics import accuracy_score
y_classes = (predictions > 0.5).astype(np.int8)

print(np.equal(y_classes, np.round(predictions)).all())

print(accuracy_score(y_classes, Y_val))

from sklearn.metrics import classification_report
print(classification_report(y_classes, Y_val))

from sklearn.metrics import precision_recall_fscore_support
precision, recall, f_score, support = precision_recall_fscore_support(Y_val, y_classes)
print(precision)
print(recall)
print(f_score)
print(support)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_val, y_classes, normalize = True)
print(accuracy)

from sklearn.metrics import precision_score
accuracy = precision_score(Y_val, y_classes)
print(accuracy)

from sklearn.metrics import recall_score
accuracy = recall_score(Y_val, y_classes)
print(accuracy)

from sklearn.metrics import f1_score
accuracy = f1_score(Y_val, y_classes)
print(accuracy)

X_test['MARRIAGE'] = np.where((X_test.MARRIAGE == 0), 3,X_test.MARRIAGE)
X_test['EDUCATION'] = np.where(((X_test.EDUCATION == 0) | (X_test.EDUCATION == 5) | (X_test.EDUCATION == 6)), 4, X_test.EDUCATION)

X_test = X_test.astype('float32')
scaler = StandardScaler()
scaler.fit(X_test)
X_test = scaler.transform(X_test)
predictions = model.predict(X_test)

y_classes = (predictions > 0.5).astype(np.int8)

np.savetxt("result.txt", y_classes,fmt='%d')