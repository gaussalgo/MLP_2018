import os
import json
import pickle
import numpy as np
from keras.preprocessing import image
from sklearn.neighbors import KDTree
from glob import glob
from tqdm import tqdm
from IPython.core.display import display
from src import is_utils

# keras future-warning supression
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    
    from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

import os, socket, json

def send_and_receive(data):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((os.environ['CDSW_MASTER_IP'], 6000))
    s.send(json.dumps(data))
    data_str = s.recv(1024)
    if data_str:
        return json.loads(data_str)
    else:
        return None
    s.close()
  

# Open a TCP connection to the master and send a request for my configuration.
# expect data in format: {u'workers': 2, u'worker_id': 1, u'folder_path': u'data/unique_1k_images/'}
cfg = send_and_receive({'worker_engine_id': os.environ['CDSW_ENGINE_ID'], 'type': 'started'})

# select my data
all_files = sorted(glob('data/unique_1k_images/*jpg'))
start_idx = len(all_files) / cfg['workers'] * cfg['worker_id']
if cfg['worker_id'] + 1 != cfg['workers']:
    end_idx = len(all_files) / cfg['workers'] * (cfg['worker_id'] + 1)
else:
    end_idx = len(all_files)

my_files = all_files[start_idx: end_idx]

print('%d: My files: %s ...' % (cfg['worker_id'], (', '.join(my_files))[:100]))
print('%d: loading model' % cfg['worker_id'])

# # Create metafeature vectors from images or skip it if present
image_vectors_file = 'data/image_vectors_%02d-%02d.pkl' % (cfg['worker_id'], cfg['workers'])
if not os.path.isfile(image_vectors_file):
    # ## Load ResNet50 neural network model
    # Do not load top classification layer but use lower layer for creating embedding
    model = ResNet50(weights='imagenet', include_top=False)

    embedding = []
    file_paths = []
    for filename in tqdm(my_files, desc='worker %d' % cfg['worker_id'], position=cfg['worker_id']):
        embedding.append(is_utils.img_path_to_metafeatures(filename, model))
        file_paths.append(filename)

    with open(image_vectors_file, 'wb') as f:
        data = {
            'embedding': embedding,
            'file_paths': file_paths,
        }
        pickle.dump(data, f)

        
# Open a TCP connection to the master and send my result.
_ = send_and_receive({'worker_id': cfg['worker_id'], 'filename': image_vectors_file, 'type': 'finished'})

print('%d: worker finished' % cfg['worker_id'])
