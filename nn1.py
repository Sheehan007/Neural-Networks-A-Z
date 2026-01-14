import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# loading data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# normalise pixel values
x_train = x_train/255.0
x_test = x_test/255.0
# flatten images
x_train = x_train.reshape(-1,784)
x_test = x_test.reshape(-1,784)
# building the neural network
model = tf.keras.Sequential([tf.keras.layers.Dense(128, activation = 'relu', input_shape = (784,),),
                             tf.keras.layers.Dense(10, activation = 'softmax')])
# compile the model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
# train the model
model.fit(x_train, y_train, epochs = 5, batch_size = 32)
# test the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print ("Test Accuracy: ", test_acc)
# predict a digit
prediction = model.predict(x_test[0].reshape(1, 784))
print("Predicted Digit: ", np.argmax(prediction))
