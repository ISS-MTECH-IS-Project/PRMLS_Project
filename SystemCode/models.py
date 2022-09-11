# SingletonMeta
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


types = ["arowana", "betta", "goldfish", "luohan", "idk"]


class Classifier(metaclass=SingletonMeta):
    def __init__(self):
        self.generalModel = None

    def classify(self, img):

        res = {}
        res["type"] = types[0]
        res["probability"] = 0.8
        return res
