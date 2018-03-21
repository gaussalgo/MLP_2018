from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np


def img_to_metafeatures(img, model):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return model.predict(x).flatten()

def img_path_to_metafeatures(img_path, model):
    target_size = (224, 224)
    img_orig = image.load_img(img_path)
    
    resized_image = img_orig.resize(target_size)

    return img_to_metafeatures(resized_image, model)
