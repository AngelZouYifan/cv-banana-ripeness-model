# import tensorflow
# import keras
from keras import models
import numpy as np
from keras.utils import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input

CLASSES = [' unripe', ' freshunripe', ' freshripe',' ripe', ' overripe',' rotten']
IMG_WIDTH, IMG_HEIGHT = 224, 224

### Load the model
json_file = open('model_structure.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = models.model_from_json(loaded_model_json)
loaded_model.load_weights('model_weights.h5')

### Preprocess single images
def preprocess(img_path):   # convert image to np array
    img = load_img(img_path, target_size = (IMG_WIDTH, IMG_HEIGHT))
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img)
    return img

def classify(img):
    pred = loaded_model.predict(img)
    name_class = CLASSES[np.argmax(pred)]
    return name_class

### Test single images
img_path = ''
img = preprocess(img_path)
name_class = classify(img)


### Test with data generator
# dir_path = 'dev_ds/train'

# from keras.preprocessing.image import ImageDataGenerator

# test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
# test_generator=test_datagen.flow_from_directory(dir_path, target_size=(224,224), color_mode='rgb', batch_size=32, classes = CLASSES, class_mode='categorical', shuffle=False)
# y_val = test_generator.classes
# pred = loaded_model.predict(test_generator)
# accuracy_TTA = np.mean(np.equal(y_val, np.argmax(pred, axis=-1)))
# print('Accuracy: {}'.format(accuracy_TTA))


### Record results in csv

# import os, fnmatch, csv

# for fd in os.listdir(dir_path):
#     if os.path.isdir(os.path.join(dir_path,fd)):
#         # cnt = 0
#         for img_path in fnmatch.filter(os.listdir(dir_path+'/'+fd),'*.jpg'):
#             # cnt+=1
#             # new_name = fd+str(cnt)+'.jpg'
#             # new_path = dir_path+'/'+fd+'/'+new_name
#             os.rename(dir_path+'/'+fd+'/'+img_path,dir_path+'/'+fd+'/'+img_path+'.jpg')


# csv_file = open('gerald_ds.csv','w')
# writer = csv.writer(csv_file, delimiter=',')
# writer.writerow(['image','class','prediction'])

# for fd in CLASSES:
#     if os.path.isdir(os.path.join(dir_path,fd)):
#         for img_name in fnmatch.filter(os.listdir(dir_path+fd),'*.jpg'):
#             img_path = dir_path+fd+'/'+img_name
#             img = preprocess(img_path)
#             name_class = classify(loaded_model, img)
#             writer.writerow([img_name,fd,name_class])

# csv_file.close()