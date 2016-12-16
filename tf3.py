
from keras.datasets import cifar10
from scipy.misc import toimage
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
import numpy as np
K.set_image_dim_ordering('th')


seed = 7
numpy.random.seed(seed)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()



X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0



y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

restore = True

if(not restore):
	model = Sequential()
	model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
	print(dir(model))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	epochs = 15
	lrate = 0.01
	decay = lrate/epochs
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	print(model.summary())
	model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=32)
	model.save('my_model.h5')


model = load_model('my_model.h5')


def lookup(place):
	if(place == 0):
		return "airplane"
	elif(place == 1):
		return "automobile"
	elif(place == 2):
		return "bird"
	elif(place == 3):
		return "cat"
	elif(place ==4):
		return "deer"
	elif(place ==5):
		return "dog"
	elif(place ==6):
		return "frog"
	elif(place ==7):
		return "horse"
	elif(place==8):
		return "ship"
	elif(place ==9):
		return "truck"

print("-----------------------------------------")
print("-----------------------------------------")
print("Prediction Number || Predicted >>> Answer")
print("+++++++++++++++++++++++++++++++++++++++++")
for i in range(40):
	index = np.random.randint(1000)
	print("         %0.2d       || %s >>> %s"%(i, \
		str(lookup(np.argmax(y_test[index]))), \
		str(lookup(np.argmax(model.predict(np.array(X_test[index]).reshape(-1,3,32,32)))))))





