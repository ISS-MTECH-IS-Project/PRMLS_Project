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
from tensorflow.keras.regularizers import l2
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
dualModelPath = "./models/compare_128_50_16_1_2022-09-18_23-10-14.hdf5"
filepath = "./models/base_128_200_64_1_2022-09-18_17-14-53.hdf5"
generalModel = "./models/medium_224_20_64_1_2022-09-25_19-33-18.hdf5"
preModel = "./models/202209251427cifar100.hdf5"

binaryModels = [
    {
        "fishType": "arawana",
        "filePath": "./models/arowana_softmax_128_50_64_1_2022-09-25_22-34-38.hdf5",
        "activation": "softmax",
        "loss": 'categorical_crossentropy',
        "optmz": 1
    },
    {
        "fishType": "betta",
        "filePath": "./models/betta_softmax_128_50_64_1_2022-09-25_22-23-48.hdf5",
        "activation": "softmax",
        "loss": 'categorical_crossentropy',
        "optmz": 1
    },
    {
        "fishType": "goldfish",
        "filePath": "./models/goldfish_softmax_128_50_64_1_2022-09-25_22-37-40.hdf5",
        "activation": "softmax",
        "loss": 'categorical_crossentropy',
        "optmz": 1
    },
    {
        "fishType": "flowerhorn",
        "filePath": "./models/luohan_softmax_128_50_64_1_2022-09-25_22-42-17.hdf5",
        "activation": "softmax",
        "loss": 'categorical_crossentropy',
        "optmz": 1
    }
]
types = ["arowana", "betta", "goldfish",
         "flowerhorn", "others", "I cannot tell"]
UPLOAD_FOLDER = "./static/images"

modelname = 'base'

IMGSIZE = 128
EPOCHS = 200
BATCH_SIZE = 64
OPT_IDX = 1
N_LABELS = 4
ACTIVATION = "softmax"
IMG_WIDTH = 224  # IMG_SIZE
IMG_HEIGHT = 160  # IMG_SIZE
CHANNELS = 3  # Keep RGB color channels to match the input format of the model

modelname = modelname+"_"+str(IMGSIZE)+"_" + \
    str(EPOCHS)+"_"+str(BATCH_SIZE)+"_"+str(OPT_IDX)
optmzs = ['adam', optimizers.RMSprop(learning_rate=0.0001), 'rmsprop']
optmz = optmzs[OPT_IDX]


def createPreModel():
    model = Sequential()
    #     73.58% - 150
    model.add(Conv2D(128, (3, 3), kernel_regularizer=l2(0.0001),
              input_shape=(32, 32, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), kernel_regularizer=l2(
        0.0001), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(512, (3, 3), kernel_regularizer=l2(
        0.0001), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(1024, (3, 3), kernel_regularizer=l2(
        0.0001), padding='same', activation='relu'))
    # model.add(Conv2D(1024, (3, 3), activation = 'relu'))
    # model.add(Conv2D(1024, (5, 5), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='softmax'))

    model.load_weights(preModel)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optmz, metrics=['accuracy'])

    return model


def createGeneralModel():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='same', input_shape=(
        IMG_HEIGHT, IMG_WIDTH, CHANNELS), kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same', kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5), padding='same', kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (4, 4), padding='same', kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (4, 4), padding='same', kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(768, (3, 3), padding='same', kernel_initializer='he_normal',
              kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Conv2D(1024, (2,2), padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(768, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    #model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(N_LABELS, activation=ACTIVATION))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optmz, metrics=['accuracy'])

    model.load_weights(generalModel)
    return model


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


def createBettaModel(params):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(
        IMGSIZE, IMGSIZE, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation=params["activation"]))

    model.load_weights(params["filePath"])
    model.compile(loss=params["loss"],
                  optimizer=optmzs[params["optmz"]], metrics=['accuracy'])
    return model


def createArowanaModel(params):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(
        IMGSIZE, IMGSIZE, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation=params["activation"]))

    model.load_weights(params["filePath"])
    model.compile(loss=params["loss"],
                  optimizer=optmzs[params["optmz"]], metrics=['accuracy'])
    return model


def createGoldfishModel(params):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(
        IMGSIZE, IMGSIZE, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation=params["activation"]))

    model.load_weights(params["filePath"])
    model.compile(loss=params["loss"],
                  optimizer=optmzs[params["optmz"]], metrics=['accuracy'])
    return model


def createFlowerhornModel(params):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(
        IMGSIZE, IMGSIZE, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation=params["activation"]))

    model.load_weights(params["filePath"])
    model.compile(loss=params["loss"],
                  optimizer=optmzs[params["optmz"]], metrics=['accuracy'])
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


def process_image(img_file, h=IMG_HEIGHT, w=IMG_WIDTH):
    # img = tf.keras.utils.load_img(
    #    img_file, target_size=(IMG_HEIGHT, IMG_WIDTH), keep_aspect_ratio=True
    # )
    img = tf.keras.utils.load_img(
        img_file, target_size=None, keep_aspect_ratio=True
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.keras.preprocessing.image.smart_resize(
        img_array, size=(h, w))
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    img_array = img_array / 255.0

    return img_array


class Classifier(metaclass=SingletonMeta):
    def __init__(self):
        self.generalModel = None
        self.modelGo = createModel()
        self.modelGo.load_weights(filepath)
        self.modelGo.compile(loss='categorical_crossentropy',
                             optimizer=optmz,
                             metrics=['accuracy'])

        self.generalModel = createGeneralModel()
        self.preModel = createPreModel()
        self.modelDualGo = createDualTwModel()
        self.modelDualGo.load_weights(dualModelPath)
        self.modelDualGo.compile(loss='binary_crossentropy',
                                 optimizer=optmz,
                                 metrics=['accuracy'])

        self.individualModels = [createArowanaModel(binaryModels[0]), createBettaModel(
            binaryModels[1]), createGoldfishModel(binaryModels[2]), createFlowerhornModel(binaryModels[3])]

    def classify(self, image):

        img_file = UPLOAD_FOLDER+"/"+image
        print(img_file)

        preRes = self.preModel.predict(process_image(img_file, 32, 32))
        print(preRes[0])
        if (preRes[0][1] != max(preRes[0])):
            resArray = [{"type": "IDK", "probability": 100*preRes[0][1]}]
            res = {"result": resArray}
            return res

        resized_img = process_image(img_file, IMGSIZE, IMGSIZE)
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

        print("individual prediction-------------------------")
        for i in range(4):
            result = self.individualModels[i].predict(resized_img)
            resArray.append(
                {"type": "individual_"+types[i], "probability": 100*result[0][1]})

        print("General prediction-------------------------")
        result = self.generalModel.predict(process_image(img_file))

        for (i, p) in enumerate(result[0]):
            resArray.append(
                {"type": "general_"+types[i], "probability": p*100})

        res = {"result": resArray}
        return res
