#importing necessary libraries
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from time import time
import os
from PIL import Image, ImageFile


import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, Conv3D, Conv2DTranspose, Input, Reshape, UpSampling2D, InputLayer, merge, concatenate
from keras.layers import Activation, Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import TensorBoard

dir_path = 'C:\\Users\\kdubelite\\Desktop\\College\\Semester 6\\Data Science\\DS Project' ## set to your main directory path
input_path = 'C:\\Users\\kdubelite\\Desktop\\College\\Semester 6\\Data Science\\DS Project\\Input' ## set to your path where you want to store input image
output_path = 'C:\\Users\\kdubelite\\Desktop\\College\\Semester 6\\Data Science\\DS Project\\output' ## set to your path where you want to store output image+9

#set the data path to the directory preceding the directory which contains the images. flow from directory takes the input in this way
data_path = 'C:\\Users\\kdubelite\\Desktop\\College\\Semester 6\\Data Science\\DS Project\\Data\\train'

train_data = ImageDataGenerator(rescale = 1. / 255)
train = train_data.flow_from_directory(data_path, target_size = (224, 224), batch_size = 32, class_mode = None)

#converting all the images to LAB colour space and normalizing
#after normalization they are converted to numpy arrays
X = []
y = []
for img in train[0]:
    try:
        lab = rgb2lab(img)
        X.append(lab[:,:,0])
        y.append(lab[:,:,1:] / 128)
    except:
        print('error')
X = np.array(X)
y = np.array(y)
X = X.reshape(X.shape+(1,))

#VGG19 model is used for as the base for transfer learning
#make instance of VGG19 mode
vggmodel = tf.keras.applications.vgg19.VGG19(weights = 'imagenet')

#Extracting the feature extraction part of VGG19 model upto the last MaxPooling Layer
#set the layers as untrainable
newmodel = Sequential() 
num = 0
for i, layer in enumerate(vggmodel.layers):
    if i<22:
        newmodel.add(layer)
#newmodel.summary()
for layer in newmodel.layers:
    layer.trainable=False
    
#predict the features from the VGG19 model, these features will be used as input for the decoder
vggfeatures = []
for i, sample in enumerate(X):
    sample = gray2rgb(sample)
    sample = sample.reshape((1,224,224,3))
    prediction = newmodel.predict(sample)
    prediction = prediction.reshape((7,7,512))
    vggfeatures.append(prediction)
vggfeatures = np.array(vggfeatures)

#Creating Input layer as VGG19 gives output of dimension (7 x 7 x 512)
encoder_input = Input(shape=(7, 7, 512,))

#Decoder
decoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_input)
decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
model = Model(inputs=encoder_input, outputs=decoder_output)
model.summary()

#run the model for 500 epochs
from keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir='C:\\Users\\kdubelite\\Desktop\\College\\Semester 6\\Data Science\\DS Project\\Logs\\')
model.compile(optimizer='adam', loss='mse' , metrics=['accuracy'])
history = model.fit(vggfeatures, y, verbose=1, epochs=500, batch_size=32)

from skimage.io import imsave
from skimage import img_as_ubyte

#define the output function which takes the test image and attaches the coloured component to it
#if the image is already coloured then it splits the image about grayscale and colored components 
def get_output():
    file = os.listdir(input_path)
    test = img_to_array(load_img(input_path + '\\' + file[0]))
    test = resize(test, (224,224), anti_aliasing=True)
    test*= 1.0/255
    lab = rgb2lab(test)
    l = lab[:,:,0]
    L = gray2rgb(l)
    L = L.reshape((1,224,224,3))
    vggpred = newmodel.predict(L)
    ab = model.predict(vggpred)
    ab = ab*128
    cur = np.zeros((224, 224, 3))
    cur[:,:,0] = l
    cur[:,:,1:] = ab
    os.chdir(output_path)
    imsave(r'out.jpg', img_as_ubyte(lab2rgb(cur)))
    os.chdir(dir_path)

#cleaning the input and output directories
def cleaning_directories():
    for input_file in os.scandir(input_path):
        os.remove(input_file.path)
    for output_file in os.scandir(output_path):
        os.remove(output_file.path)