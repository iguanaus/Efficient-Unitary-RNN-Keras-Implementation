from __future__ import print_function
from keras import backend as K
import os
import keras
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Recurrent, SimpleRNN, LSTM
import numpy as np
from EURNN import *
import tensorflow as tf

print(keras.backend.backend())


init = tf.global_variables_initializer()
#sess.run(init)
trainfraction=0.8
history=100
d = 1

print("Loading sleep data...")
h5f = h5py.File('data/sleep_wire1.h5', 'r')
Z = h5f['data'][:]
h5f.close()
#Z = Z[:1000000]
n = Z.shape[0]
labels = Z.shape[1]-d
batches = int(n/history)
trainbatches = int(trainfraction*batches)
n = batches*history

print("Randomly shuffling snippets...")
Z=Z[:n].reshape(batches,history,d+labels)
np.random.shuffle(Z)

# X = n vectors of length d=1
# Y = n vectors of length labels
# The RNN expects the input data (X) to be provided with a specific array structure in the form of: [batches,history,vector length].
X = Z[:,:,0].reshape((Z.shape[0],Z.shape[1],1))
Y = Z[:,:,1:]

# Train:
x_train = X[:trainbatches].astype('float32')
y_train = Y[:trainbatches].astype('float32')
x_test = X[trainbatches:].astype('float32')
y_test = Y[trainbatches:].astype('float32')

#x_train = 0*x_train  # Kill input to evaluate baseline performance

model = Sequential()
#model.add(SimpleRNN(40,input_shape=(history,d),activation='relu',return_sequences='true')) 
model.add(SimpleRNN(40,input_shape=(history,d),activation='relu',return_sequences='true')) 
#,kernel_initializer='lecun_uniform',recurrent_initializer='orthogonal'
#The test is to get this to work
model.add(EURNNCell(80,return_sequences='true'))
model.add(EURNNCell(80,return_sequences='true'))
#model.add(EURNNCell(40,input_shape=(history,d),return_sequences='true'))
#model.add(EURNNCell(40,input_shape=(history,d),return_sequences='true'))



#model.add(LSTM(10,input_shape=(history,d),activation='relu',return_sequences='true',kernel_initializer='lecun_uniform',recurrent_initializer='orthogonal'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(labels, activation='softmax'))
#model.compile()
#model.compile(loss=tf.nn.sparse_softmax_cross_entropy_with_logits,optimizer=tf.train.RMSPropOptimizer,metrics=['accuracy'])
print(K.backend())
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.adam(lr=0.01),metrics=['accuracy'])
print("Model compiled and running....")
hist = model.fit(x_train, y_train, validation_split=0.2,batch_size=100,epochs=1)
print("Model done running")

# Export losses to file:
key = hist.history.keys()
epochs = len(hist.history[key[0]])
loss = np.zeros(4*epochs).reshape(4,epochs)
for i in range(0,4):
    loss[i]=np.array(hist.history[key[i]])

np.savetxt('output/losses.dat',np.transpose(loss))

model.save_weights("output/sleepRNN.h5")


score = model.evaluate(x_test, y_test, verbose=0)
#score = model.evaluate(0*x_test, y_test, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

yhat = model.predict(x_test)
ntests = y_test.shape[0]*y_test.shape[1]
C = np.dot(np.transpose(y_test.reshape(ntests,labels)),yhat.reshape(ntests,labels))/ntests
np.savetxt('output/Crnn.dat',C)


print("ALL DONE: Ignore everything below.")