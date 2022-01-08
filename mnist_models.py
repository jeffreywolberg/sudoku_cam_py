from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model as kerasModel


def MNIST_medium_model() -> kerasModel:
    mod = Sequential()
    mod.add(Conv2D(30, (5, 5), input_shape=(40, 40, 1), activation='relu'))
    mod.add(MaxPooling2D(pool_size=(2, 2)))
    mod.add(Conv2D(15, (3, 3), activation='relu'))
    mod.add(MaxPooling2D(pool_size=(2, 2)))
    mod.add(Dropout(0.2))
    mod.add(Flatten())
    mod.add(Dense(128, activation='relu'))
    mod.add(Dense(50, activation='relu'))
    mod.add(Dense(10, activation='softmax'))
    # Compile model
    mod.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return mod


def MNIST_large_model() -> kerasModel:
    mod = Sequential()
    mod.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(34, 34, 1)))
    mod.add(Conv2D(32, (5, 5), activation='relu'))
    mod.add(MaxPooling2D((2, 2)))
    mod.add(Dropout(0.20))

    mod.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    mod.add(Conv2D(64, (3, 3), activation='relu'))
    mod.add(MaxPooling2D((2, 2)))
    mod.add(Dropout(0.20))

    mod.add(Flatten())
    mod.add(Dense(128, activation='relu'))
    mod.add(Dense(50, activation='relu'))
    mod.add(Dense(10, activation='softmax'))
    mod.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return mod

def MNIST_large_model2(im_length=28) -> kerasModel:
    weight_decay = 1e-3
    mod = Sequential()
    mod.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(im_length, im_length, 1),
                   ))
    mod.add(MaxPooling2D((2, 2)))
    mod.add(Conv2D(32, (5, 5), activation='relu'))
    mod.add(Dropout(0.20))

    mod.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    mod.add(MaxPooling2D((2, 2)))
    mod.add(Conv2D(64, (3, 3), activation='relu'))
    mod.add(Dropout(0.20))

    mod.add(Flatten())
    mod.add(Dense(128, activation='relu'))
    mod.add(Dense(50, activation='relu'))
    mod.add(Dense(10, activation='softmax'))
    mod.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return mod