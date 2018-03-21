import multiprocessing
import os
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, ResNet50

model = ResNet50(weights='imagenet', include_top=False)

# ## Simple functions for transforming images into feature-vectors
def img_to_metafeatures(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x).flatten()

def img_path_to_metafeatures(img_path):
    target_size = (224, 224)
    img_orig = image.load_img(img_path)
    resized_image = img_orig.resize(target_size)
    return img_to_metafeatures(resized_image)

#def worker1(num):
#    print('ok')
  
def worker1(num):
    img_path='data/unique_1k_images/'
    embedding = []
    file_paths = []
    for filename in os.listdir(img_path):
        embedding.append(img_path_to_metafeatures(os.path.join(img_path, filename)))
        file_paths.append(filename)
    print('ok')
  
if __name__ == '__main__':
  jobs = []
  for i in range(5):
    p = multiprocessing.Process(target=worker1, args=(i,))
    jobs.append(p)
    p.start()