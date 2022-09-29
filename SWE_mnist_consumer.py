#%%
import numpy as np
from keras.datasets import mnist
from keras.models import load_model

# Loading Model
model = load_model('./MNIST_NN.h5')

# Loading Image
#for now take standard test to see if model loaded
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize again
x_test = x_test.astype('float32')
x_test = x_test / 255.0
x_test = x_test.reshape(x_test.shape[0], -1)

#test to see model perform
print("\nChecking raw data prediction of first test sample from dataset: ")
pred = model.predict(x_test[0:1])
pred_classes = np.argmax(pred, axis=1)
print("Prediction array for x_text[0] = {}".format(pred[0])) # Array of statistical probability of each class for test data[0]
print("Predicted number: {}\nWith {}% confidence.".format(pred_classes[0], 100*pred[0][pred_classes[0]]))

