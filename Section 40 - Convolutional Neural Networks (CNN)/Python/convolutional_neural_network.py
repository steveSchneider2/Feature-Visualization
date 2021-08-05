'''
Steve Schneider, 04 Feb 2021;
This file is from "Deep Learning A-Z: Hands-on artificial neural networks"
Taught by Kirill Eremenko (Ukrainian?) and Hadelin De Ponteves
(Udemy course)

'''
# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
import os
import time
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tf.__version__
pver = str(format(sys.version_info.major) +'.'+ format(sys.version_info.minor)+'.'+ format(sys.version_info.micro))

condaenv = os.environ['CONDA_DEFAULT_ENV']

starttime = time.perf_counter()
modelstart = time.strftime('%c')

#%% Part 1 - Data Preprocessing

#%% Preprocessing the Training set... Image Augmentation
train_datagen = ImageDataGenerator(rescale = 1./255,    # get all the pixel values between 0 & 1.
                                   shear_range = 0.2,   # augmentation
                                   zoom_range = 0.2,    # more augmentation
                                   horizontal_flip = True)  # and yet more
training_set = train_datagen.flow_from_directory('../dataset/training_set',
                                                 target_size = (64, 64),  # was originally 150,150... but this takes a LONG time to train.
                                                 batch_size = 32,         # 64 (above, was chose arbitrarily to be smaller)  
                                                 class_mode = 'binary')

#%% Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)  # get all the pixel values between 0 & 1.
test_set = test_datagen.flow_from_directory('../dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

#%% Part 2 - Building the CNN

#%% Initialising the CNN ... which is a 'sequence of layers'...therefore use a sequence...
#      Library, module, class
cnn = tf.keras.models.Sequential()

#%% Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32,  # filters are the feature detectors;
                               kernel_size=3,  # Size of each filter: 3 by 3
                               activation='relu',  # the Rectifier activation
                               input_shape=[64, 64, 3])) # from the target size that we set above on line 38; b/w would be 1, but we are dealing with  color... 3

#%% Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#%% Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#%% Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

#%% Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

#%% Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#%% Part 3 - Training the CNN

#%% Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#%% Training the CNN on the Training set and evaluating it on the Test set
epochs = 25
# cnn.fit(x = tt,)
mdl = cnn.fit(x = training_set, validation_data = test_set, epochs = epochs)

endtime = time.perf_counter()
duration = round(endtime - starttime,2)
#%% Graph the model
import pydot
import graphviz
from tensorflow import keras
# from tensorflow.keras import layers
keras.utils.plot_model(cnn, show_shapes=True)

cnn.summary()
import io
s = io.StringIO()
cnn.summary(print_fn=lambda x: s.write(x + '\n'))
model_summary = s.getvalue()
model_summary = model_summary.replace('=','')
model_summary = model_summary.replace('_','')
model_summary = model_summary.replace('\n\n','\n')

#%% Plot loss per iteration  & Plot accuracy per iteration
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
fig, ax = plt.subplots(figsize=(12, 8))
trans = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
plt.style.use('classic')
plt.plot(mdl.history['loss'], label='loss', linewidth=2)
plt.plot(mdl.history['val_loss'], label='val_loss', linewidth=2)
plt.legend()

plt.plot(mdl.history['accuracy'], label='acc', linewidth=3)
plt.plot(mdl.history['val_accuracy'], label='val_acc', linewidth=3)
plt.legend()
plt.title(f'Udemy DeepLearning Lecture 41 {modelstart}\n ' +
          f'Image Classification: (DOGS & CATS) 8000 records ' +
          f'unk size')
plt.xlabel(f'Convolutional NN  Duration: {duration}')
# plt.ylim(0,1)
xloc=.05
plt.text(xloc, .7,# transform=trans1,
         s='This "architecture" of 32 feature detectors in each of the' +
         ' first two layers is "Classic"', 
         wrap=True, ha='left', va='bottom', transform=trans,
         fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
plt.text(xloc, .2,# transform=trans1,
         s=model_summary,transform=trans,
         wrap=True, ha='left', va='bottom', fontname='Consolas',
         fontsize=12, bbox=dict(facecolor='pink', alpha=0.5))
plt.text(xloc, .02,# transform=trans1,
         s=f'Conda Envr:  {condaenv}' +
         f'Gpu  Support:       {tf.test.is_built_with_gpu_support()}\n' +
         f'Python: {pver} tensorflow Ver: {tf.version.VERSION}' +
         '\nConvolutional neural network' + 
         f'\n{epochs} epochs, Duration: {duration:3.2f} seconds',# +
#         f'\n{gpus[0].name} Cuda 11.1.relgpu',
         wrap=True, ha='left', va='bottom',transform=trans,
         fontsize=12, bbox=dict(facecolor='aqua', alpha=0.5))
plt.show()

#%% Part 4 - Making a single prediction

import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('../dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)