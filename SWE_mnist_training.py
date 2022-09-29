#%%
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense

# Load public data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = 10


# Prepare Data
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#Dimension Reduction
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
#Colorgrade Reduction
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0


# Define model
model = Sequential()
model.add(Dense(units=128, input_shape=(784,), activation='relu'))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# Train Model
model.fit(x_train, y_train, batch_size=512, validation_data=(x_test, y_test), epochs=10)

print("MODEL CREATED.\nPlease use SWE_mnist_consumer.py to predict numbers.")
# Save the Model
model.save('./MNIST_NN.h5')

# %%
