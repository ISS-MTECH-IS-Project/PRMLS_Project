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
from keras.models import load_model
# SingletonMeta


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


preModel = "./models/Hybrid_Cifar100_202210062242.hdf5"
finalModel = "./models/arbitrator_softmax_50_32_1_2022-10-08_00-05-20.hdf5"
members_param = [
    {
        "model_name": "general",
        "file_path": "./models/medium_224_20_64_1_2022-09-25_19-33-18.hdf5",
        "class_names": ['arowana', 'betta', 'goldfish', 'flowerhorn'],
        "img_width": 224,
        "img_height":160
    },
    {
        "model_name": "arawana",
        "file_path": "./models/arowana_softmax_128_50_64_1_2022-09-25_22-34-38.hdf5",
        "class_names": ['not arowana', 'arowana'],
        "img_width": 128,
        "img_height":128
    },
    {
        "model_name": "betta",
        "file_path": "./models/betta_softmax_128_50_64_1_2022-09-26_17-51-39.hdf5",
        "class_names": ['not betta', 'betta'],
        "img_width": 128,
        "img_height":128
    },
    {
        "model_name": "goldfish",
        "file_path": "./models/goldfish_softmax_128_50_64_1_2022-09-25_22-37-40.hdf5",
        "class_names": ['not goldfish', 'goldfish'],
        "img_width": 128,
        "img_height":128
    },
    {
        "model_name": "flowerhorn",
        "file_path": "./models/luohan_softmax_128_50_64_1_2022-09-25_22-42-17.hdf5",
        "class_names": ['not flowerhorn', 'flowerhorn'],
        "img_width": 128,
        "img_height":128
    }
]
UPLOAD_FOLDER = "./static/images"

CHANNELS = 3  # Keep RGB color channels to match the input format of the model


def preprocess_image(filename):
    images = []
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
    for i, m in enumerate(members_param):
        image_resized = tf.image.resize_with_pad(
            image_decoded, m["img_height"], m["img_width"], antialias=False)
        image_normalized = image_resized / 255.0
        images.append(np.expand_dims(image_normalized.numpy(), 0))
    return images

# load models from file


def load_all_models():
    all_models = list()

    for m in members_param:
        # define filename for this ensemble
        filename = m["file_path"]
        # load model from file
        model = load_model(filename)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


def process_image(img_file, h, w):
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
        self.preModel = load_model(preModel)
        self.finalModel = load_model(finalModel)
        self.models = load_all_models()

    def classify(self, image):
        img_file = UPLOAD_FOLDER+"/"+image
        print(img_file)

        print("fish or not fish classification-----------------")
        preRes = self.preModel.predict(process_image(img_file, 64, 64))
        print(preRes[0])
        if (np.argmax(preRes[0]) == 1):
            resArray = [{"type": "I cannot tell if it is a fish.",
                         "probability": 100*np.max(preRes[0])}]
            res = {"result": resArray}
            return res

        resArray = []
        input_images = preprocess_image(img_file)
        print("ensemble prediction ------------------------------")
        predict = self.finalModel.predict(input_images)
        print(predict[0])
        fish_type = members_param[0]["class_names"][np.argmax(predict[0])]
        score = 100*np.max(predict[0])
        resArray.append({"type": fish_type, "probability": score})

        for i, model in enumerate(self.models):
            model_name = members_param[i]["model_name"]
            print(model_name+" prediction ------------------------")
            img = input_images[i]
            print(np.shape(img))
            predict = model.predict(img)
            fish_type = members_param[i]["class_names"][np.argmax(predict[0])]
            score = 100*np.max(predict[0])
            resArray.append({"type": fish_type, "probability": score})

        res = {"result": resArray}
        return res
