# -*- coding: utf-8 -*-

# Import Dependencies
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import keras
from keras.layers import Dense,GlobalAveragePooling2D
from keras import applications
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, MobileNet
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

# Properties and hyperparameters
CLASSES = [' freshripe',' freshunripe',' overripe',' ripe',' rotten',' unripe']

lr = 0.0003
arch = 'mb'
ds = 'dev_ds2'

# Preprocess data
# A DirectoryIterator yielding tuples of (x, y) where x is a numpy array containing a batch of images with shape (batch_size, *target_size, channels) and y is a numpy array of corresponding labels.
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator=train_datagen.flow_from_directory(ds+'/train', target_size=(224,224), color_mode='rgb', classes = CLASSES, batch_size=32, class_mode='categorical', shuffle=True)  # train

valid_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
valid_generator=train_datagen.flow_from_directory(ds+'/valid', target_size=(224,224), color_mode='rgb', classes = CLASSES, batch_size=32, class_mode='categorical', shuffle=True)  # valid

test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator=train_datagen.flow_from_directory(ds+'/test', target_size=(224,224), color_mode='rgb', classes = CLASSES,batch_size=32,  class_mode='categorical', shuffle=True)  # test

# Build the model
base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(len(CLASSES),activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)
#specify the inputs
#specify the outputs
#now a model has been created based on our architecture

for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True

model.compile(optimizer=Adam(lr),loss='categorical_crossentropy',metrics=['accuracy'])

# Train the model
hs_train = model.fit(train_generator, epochs=20, validation_data=valid_generator)

# Fine tuning the model
base_model.trainable = True
model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy']) # slow learning rate
hs_tune = model.fit(train_generator, epochs=20, validation_data=valid_generator)

# Test
score = model.evaluate(test_generator)

# Plot
def plot(hs): 
  # Plot accuracy
    plt.plot(hs.history['accuracy'])
    plt.plot(hs.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot loss
    plt.plot(hs.history['loss'])
    plt.plot(hs.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()

plot(hs_train)
plot(hs_tune)


model.save(arch+'_'+ds+'_'+str(lr)+'.h5')

from keras.models import model_from_json
model_json = loaded_model.to_json()
with open("model_structure.json", "w") as json_file:
    json_file.write(model_json)
loaded_model.save_weights("model_weights.h5")