import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
# SingletonMeta


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


filepath = "./models/jeremy_2022-09-13_20-51-43.hdf5"
types = ["arowana", "betta", "goldfish", "luohan", "others", "I cannot tell"]
UPLOAD_FOLDER = "./static/images"


def createModel():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(180, 180, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(40, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model
 # This is used for final testing


class Classifier(metaclass=SingletonMeta):
    def __init__(self):
        self.generalModel = None
        self.modelGo = createModel()
        self.modelGo.load_weights(filepath)
        self.modelGo.compile(loss='categorical_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy'])

    def classify(self, image):

        print(UPLOAD_FOLDER+"/"+image)
        img = cv2.imread(str(UPLOAD_FOLDER+"/"+image))
        resized_img = cv2.resize(img, (180, 180))
        resized_img = np.array([resized_img])
        resized_img = resized_img / 255
        result = self.modelGo.predict(resized_img)
        print(result)
        resArray = [{"type": types[i], "probability": p*100}
                    for (i, p) in enumerate(result[0])]
        # res["type"] = types[0]
        # res["probability"] = 0.8
        res = {"result": resArray}
        return res
