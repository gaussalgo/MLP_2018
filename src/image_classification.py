# Simple image clasification with pretrained model
# ---
# first run !pip3 install -r requirements.txt

# catch some keras's future warnings 
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    
    from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
    
from keras.preprocessing import image
import numpy as np
from IPython.core.display import display

# load resnet model (downloads data from repository)
model = ResNet50(weights='imagenet')

# Expected image size
model.input_shape

# Helper functions for image-calssification
def classify_image(img, info):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    
    print('Image: %s' % info)
    display(img)
    for class_id, tag, probability in decode_predictions(preds, top=3)[0]:
        print('%5.2f %%: %s' % (probability * 100, tag))
    print()
        
def load_and_classify(img_path):
    target_size = (224, 224)

    img_orig = image.load_img(img_path)

    image_rect_size = min(img_orig.size)
    cropped_image = img_orig.crop(box=(0, 0, image_rect_size, image_rect_size)).resize(target_size)
    
    resized_image = img_orig.resize(target_size)

    print('Original file:')
    display(img_orig)
    print
    
    classify_image(resized_image, 'Resized to %s' % str(target_size))
    classify_image(cropped_image, 'Cropped to %s' % str(target_size))

# Example predictions
load_and_classify(img_path = 'data/unique_1k_images/000001.jpg')

# Example predictions
load_and_classify(img_path = 'data/unique_1k_images/000002.jpg')