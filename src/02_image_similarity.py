# # Image similarity
# - Use the same pretrained neural network as in the previous task
# - Instead of classification use network to transform image into a vector of descriptive metafeatures
# - Property: the closer vectors the more similar images
# - Transform a set of images and index them for similarity queries
# - Use the structure for finding similar images

# imports
import os
import pickle
import numpy as np
from keras.preprocessing import image
# k-dimensional tree for spatial indexing
from sklearn.neighbors import KDTree
from glob import glob
# progress bar
from tqdm import tqdm
from IPython.core.display import display
from src import is_utils

# keras future-warning supression
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    
    from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# ## Load ResNet50 neural network model
# Do not load top classification layer but use lower layer for creating embedding
model = ResNet50(weights='imagenet', include_top=False)

# ## Extract metafeatures from image - example 
img_path='data/unique_1k_images/000001.jpg'
display(image.load_img(img_path))
metafeatures_test = is_utils.img_path_to_metafeatures(img_path, model)
print('Metafeature count: %s' % metafeatures_test.shape)
print('Metafeatures: %s' %metafeatures_test)

# # Create metafeature vectors from images or load it if present
image_vectors_file = 'data/image_vectors.pkl'
if not os.path.isfile(image_vectors_file):
    embedding = []
    file_paths = []
    for filename in tqdm(glob('data/unique_1k_images/*jpg')):
        embedding.append(is_utils.img_path_to_metafeatures(filename, model))
        file_paths.append(filename)

    with open(image_vectors_file, 'wb') as f:
        data = {
            'embedding': embedding,
            'file_paths': file_paths,
        }
        pickle.dump(data, f)
else:
    with open(image_vectors_file, 'rb') as f:
        data = pickle.load(f)
  
# ## Store into spatial database
kdtree = KDTree(data['embedding'], leaf_size=5, metric='euclidean')

# ## helper functions for finding and showing similar images
def show_img_path(image_path):
    img = image.load_img(image_path)
    img.thumbnail(size=(300, 300))
    display(img)
    
def find_similar_images(image_path, n=5):
    query = is_utils.img_path_to_metafeatures(image_path, model).reshape(1, -1)
    distances, similar_ids = kdtree.query(X=query, k=n, return_distance=True)

    print('Query image : %s' % image_path)
    show_img_path(image_path)
    for distance, image_idx in zip(distances[0], similar_ids[0]):
        if data['file_paths'][image_idx] != image_path:
            print('Image idx: %s' % image_idx)
            filename = data['file_paths'][image_idx]
            print('Distance: %8.4f' % distance)
            print('File: %s' % filename)
            show_img_path(filename)


### Find top 5 closest images
find_similar_images('data/unique_1k_images/000088.jpg', n=5)
