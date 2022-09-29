#%%
import numpy as np
import random
import sys
import cv2 
from matplotlib import pyplot
from keras.datasets import mnist
from keras.models import load_model

# Loading Model
model = load_model('./MNIST_NN.h5')
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if len(sys.argv) > 1:
    
    # Load Image File
    imgfile = sys.argv[1]
    test_image = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)

    # Format Image
    img_resized = cv2.resize(test_image, (28, 28), interpolation=cv2.INTER_LINEAR)
    x_test[0] = img_resized


    # Normalize again
    x_test = x_test.astype('float32')
    x_test = x_test / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1)


    # Predict number
    pred = model.predict(x_test[0:1])
    pred_classes = np.argmax(pred, axis=1)
    print("\n------ Classifying Loaded Sample Image ------------------------------\n")
    print("Prediction array for sample = \n{}".format(pred[0]))
    print("Predicted number: {}\nWith {}% confidence.".format(pred_classes[0], 100*pred[0][pred_classes[0]]))
    print("\n--------------------------------------------------------------")
    pyplot.imshow(test_image, cmap='gray')
    pyplot.show()

else:
    randSamp = random.randrange(1,len(x_test))
    randSampImg = x_test[randSamp]


    # Normalize again
    x_test = x_test.astype('float32')
    x_test = x_test / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1)


    # Predict number
    print("\nNo file passed\n---- Loading Random Image from MNIST test dataset ------------------\n")
    pred = model.predict(x_test[randSamp:randSamp+1])
    pred_classes = np.argmax(pred, axis=1)
    print("Prediction array for sample = \n{}".format(pred[0])) # Array of statistical probability of each class for test data[0]
    print("Predicted number: {}\nWith {}% confidence.".format(pred_classes[0], 100*pred[0][pred_classes[0]]))
    print("\n--------------------------------------------------------------")
    print("You can also try and use your own image, loading it by calling:\nSWE_mnist_consuper.py pathtoyourimage.png")
    pyplot.imshow(randSampImg, cmap=pyplot.get_cmap('gray'))
    pyplot.show()
# %%
