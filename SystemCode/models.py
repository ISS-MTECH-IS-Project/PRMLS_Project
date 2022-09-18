import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
# SingletonMeta


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


sampleFiles = ['/sample/arowana.jpg', '/sample/betta.jpg',
               '/sample/goldfish.jpg', '/sample/luohan.jpg']
dualModelPath = "./models/compare_128_12_16_0_2022-09-18_22-50-58.hdf5"
filepath = "./models/base_128_200_64_1_2022-09-18_17-14-53.hdf5"
types = ["arowana", "betta", "goldfish", "luohan", "others", "I cannot tell"]
UPLOAD_FOLDER = "./static/images"

modelname = 'base'

IMGSIZE = 128
EPOCHS = 200
BATCH_SIXE = 64
OPT_IDX = 1

modelname = modelname+"_"+str(IMGSIZE)+"_" + \
    str(EPOCHS)+"_"+str(BATCH_SIXE)+"_"+str(OPT_IDX)
optmzs = ['adam', optimizers.RMSprop(learning_rate=0.0001), 'rmsprop']
optmz = optmzs[OPT_IDX]


def createModel():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding='same', input_shape=(
        IMGSIZE, IMGSIZE, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optmz, metrics=['accuracy'])
    return model


def createDualTwModel():
    Lin = Input(shape=(IMGSIZE, IMGSIZE, 3))
    Lx = Conv2D(64, (3, 3), padding='same', activation='relu')(Lin)
    Rin = Input(shape=(IMGSIZE, IMGSIZE, 3))
    Rx = Conv2D(64, (3, 3), padding='same', activation='relu')(Rin)

    shared = Conv2D(32, (3, 3), padding='same',
                    activation='relu', name='SharedLyr')
    Lx = shared(Lx)
    Rx = shared(Rx)

    Lx = MaxPooling2D(pool_size=(2, 2))(Lx)
    Rx = MaxPooling2D(pool_size=(2, 2))(Rx)

    shared2 = Conv2D(32, (3, 3), padding='same',
                     activation='relu', name='SharedLyr2')
    Lx = shared2(Lx)
    Rx = shared2(Rx)

    Lx = MaxPooling2D(pool_size=(2, 2))(Lx)
    Rx = MaxPooling2D(pool_size=(2, 2))(Rx)

    shared3 = Conv2D(16, (3, 3), padding='same',
                     activation='relu', name='SharedLyr3')
    Lx = shared3(Lx)
    Rx = shared3(Rx)

    Lx = MaxPooling2D(pool_size=(2, 2))(Lx)
    Rx = MaxPooling2D(pool_size=(2, 2))(Rx)

    x = concatenate([Lx, Rx], axis=-1)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[Lin, Rin], outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer=optmz,
                  metrics=['accuracy'])

    return model


class Classifier(metaclass=SingletonMeta):
    def __init__(self):
        self.generalModel = None
        self.modelGo = createModel()
        self.modelGo.load_weights(filepath)
        self.modelGo.compile(loss='categorical_crossentropy',
                             optimizer=optmz,
                             metrics=['accuracy'])

        self.modelDualGo = createDualTwModel()
        self.modelDualGo.compile(loss='binary_crossentropy',
                                 optimizer=optmz,
                                 metrics=['accuracy'])

    def classify(self, image):

        print(UPLOAD_FOLDER+"/"+image)
        img = cv2.imread(str(UPLOAD_FOLDER+"/"+image))
        resized_img = cv2.resize(img, (IMGSIZE, IMGSIZE))
        resized_img = np.array([resized_img])
        resized_img = resized_img / 255
        result = self.modelGo.predict(resized_img)
        print(result)
        resArray = [{"type": types[i], "probability": p*100}
                    for (i, p) in enumerate(result[0])]

        print("Compare prediction-------------------------")
        for i in range(len(sampleFiles)):
            sample_img = cv2.imread(str(UPLOAD_FOLDER+sampleFiles[i]))
            sample_img = cv2.resize(sample_img, (IMGSIZE, IMGSIZE))
            sample_img = np.array([sample_img])
            sample_img = sample_img / 255
            result = self.modelDualGo.predict([resized_img, sample_img])
            print(sampleFiles[i], " ------- ", result)
            resArray.append(
                {"type": "dual_"+types[i], "probability": 100*result[0][0]})

        res = {"result": resArray}
        return res
