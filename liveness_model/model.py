from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
 

#input 64x64
class LivenessNet:
	@staticmethod
	def build(width, height, depth, classes):
		model = Sequential()
		input_shape = (height, width, depth)
		chan_dim = -1

		if K.image_data_format() == "channels_first":
			input_shape = (depth, height, width)
			chan_dim = 1
		
		
		model.add(Conv2D(18, (3,3), padding="same", input_shape=input_shape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chan_dim))

		model.add(Conv2D(18, (3,3), padding="same", input_shape=input_shape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(BatchNormalization(axis=chan_dim))

		model.add(Conv2D(36, (3,3), padding="same", input_shape=input_shape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chan_dim))

		model.add(Conv2D(36, (3,3), padding="same", input_shape=input_shape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(BatchNormalization(axis=chan_dim))

		model.add(Flatten())
		model.add(Dense(128))
		model.add(Activation("relu"))
		model.add(Dropout(0.7))

		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model