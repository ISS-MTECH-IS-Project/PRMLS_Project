import numpy as np
import tensorflow as tf
from keras.models import load_model
# SingletonMeta


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


preModel = "./models/Hybrid_best.hdf5"
finalModel = "./models/ensemble_best.hdf5"
members_param = [
    {
        "model_name": "Main Classification Model",
        "file_path": "./models/medium_best.hdf5",
        "class_names": ['Arowana', 'Betta', 'Goldfish', 'Flowerhorn'],
        "img_width": 224,
        "img_height":160
    },
    {
        "model_name": "Simple Classification Model",
        "file_path": "./models/simple_best.hdf5",
        "class_names": ['Arowana', 'Betta', 'Goldfish', 'Flowerhorn'],
        "img_width": 160,
        "img_height":160
    },
    {
        "model_name": "Grayscale Model",
        "file_path": "./models/grayscale_best.hdf5",
        "class_names": ['Arowana', 'Betta', 'Goldfish', 'Flowerhorn'],
        "img_width": 80,
        "img_height":60,
        "channels":1
    }
]
UPLOAD_FOLDER = "./static/images"

CHANNELS = 3  # Keep RGB color channels to match the input format of the model


def preprocess_image(filename):
    images = []

    for i, m in enumerate(members_param):
        grayscale = m.get("channels", 3) == 1
        img = tf.keras.utils.load_img(
            filename, target_size=None, keep_aspect_ratio=True, grayscale=grayscale
        )
        img_array = tf.keras.utils.img_to_array(img)
        image_resized = tf.keras.preprocessing.image.smart_resize(
            img_array, size=(m["img_height"], m["img_width"]))
        image_normalized = image_resized / 255.0
        images.append(np.expand_dims(image_normalized, 0))
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

        resArray = []
        print("fish or not fish classification-----------------")
        preRes = self.preModel.predict(process_image(img_file, 64, 64))
        print(preRes[0])
        if (np.argmax(preRes[0]) == 1):
            resArray.append({"model_name": "Fish/Not Fish Model", "type": "Not Fish",
                             "probability": 100*np.max(preRes[0])})
            res = {"result": resArray}
            return res
        else:
            resArray.append({"model_name": "Fish/Not Fish Model", "type": "Fish",
                             "probability": 100*np.max(preRes[0])})

        input_images = preprocess_image(img_file)

        for i, model in enumerate(self.models):
            model_name = members_param[i]["model_name"]
            print(model_name+" prediction ------------------------")
            img = input_images[i]
            print(np.shape(img))
            predict = model.predict(img)
            fish_type = members_param[i]["class_names"][np.argmax(predict[0])]
            score = 100*np.max(predict[0])
            resArray.append({"model_name": model_name,
                            "type": fish_type, "probability": score})

        print("ensemble prediction ------------------------------")
        predict = self.finalModel.predict(input_images)
        print(predict[0])
        fish_type = members_param[0]["class_names"][np.argmax(predict[0])]
        score = 100*np.max(predict[0])
        resArray.append({"model_name": "Ensemble Model",
                        "type": fish_type, "probability": score})

        res = {"result": resArray}
        return res
