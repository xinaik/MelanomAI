import numpy as np 
import pandas as pd 
import os
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import io
import sklearn.model_selection as skl
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50 
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, save_model, load_model
df = pd.read_csv('C:/Users/anish_8d0sbo2/Hackathon5/ISIC_2019_Training_GroundTruth.csv')
list1 = []
for index, row in df.iterrows():
    for i in df.columns:
        if row[i] == 1.0:
            list1.append(i)
df['class'] = list1

del df['MEL']
del df['NV']
del df['BCC']
del df['AK']
del df['BKL']
del df['DF']
del df['VASC']
del df['SCC']
del df['UNK']

print(df.head(50))
train_df, valid_df = skl.train_test_split(df, test_size = 0.2,)

train_idg = ImageDataGenerator(rescale=1. / 255.0,horizontal_flip = True, vertical_flip = False, height_shift_range= 0.1, width_shift_range=0.1, 
rotation_range=20, shear_range = 0.1,zoom_range=0.1)

train_gen = train_idg.flow_from_dataframe(dataframe=train_df, directory='C:/Users/anish_8d0sbo2/Hackathon5/ISIC_2019_Training_Input/ISIC_2019_Training_Input', x_col = 'img_path', y_col = 'class',class_mode = 'categorical',target_size = (224, 224), 
batch_size = 9)

val_idg = ImageDataGenerator(rescale=1. / 255.0)

val_gen = val_idg.flow_from_dataframe(dataframe=valid_df,directory='C:/Users/anish_8d0sbo2/Hackathon5/ISIC_2019_Training_Input/ISIC_2019_Training_Input', x_col = 'img_path',y_col = 'class',class_mode = 'categorical',target_size = (224, 224), 
batch_size = 6)
valX, valY = val_gen.next()
fig, m_axs = plt.subplots(5,4, figsize = (16, 16))
m_axs = m_axs.flatten()
imgs = 'C:/Users/anish_8d0sbo2/Hackathon5/ISIC_2019_Training_Input/ISIC_2019_Training_Input/'+ df.image + '.jpg'
ind=0

for img, ax in zip(imgs, m_axs):
    img = io.imread(img)
    ax.imshow(img,cmap='gray')
    ax.set_title(df.iloc[ind]['class'])
    ind=ind+1

model = VGG16(include_top=True, weights='imagenet')
transfer_layer = model.get_layer('block5_pool')
vgg_model = Model(inputs=model.input,outputs=transfer_layer.output)
for layer in vgg_model.layers[0:17]:
    layer.trainable = False
new_model = Sequential()
new_model.add(vgg_model)
new_model.add(Flatten())
new_model.add(Dense(1024, activation='relu'))
new_model.add(Dropout(0.2))
new_model.add(Dense(512, activation='relu'))
new_model.add(Dropout(0.2))
new_model.add(Dense(8, activation='sigmoid'))
optimizer = Adam(lr=1e-5)
loss = 'CategoricalCrossentropy'
metrics = ['binary_accuracy']
print(new_model.summary())
new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
history = new_model.fit_generator(train_gen, validation_data = (valX, valY), epochs = 5, steps_per_epoch=16)
filepath = './saved_model'
save_model(model, filepath, overwrite=True)