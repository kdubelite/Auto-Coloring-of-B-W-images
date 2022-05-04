# Auto-Coloring-of-B/W-images

In general, colouring a grayscale image is a straightforward process for the human mind; we learn to fill in the mission colours in colouring books from an early age by recalling that the grass is green, the sky is blue, and white clouds are there, or that an apple might be red or green. With the introduction of deep Convolutional neural networks, the challenge has gotten a lot of attention as a representative issue for a thorough visual comprehension of artificial intelligence, akin to what many people previously thought object recognition was.

In this project, I have used the encoder-decoder architecture with concept of transfer learning for auto colourizatoin of black and white images or any image in general. There are various other advanced architectures already in use for this problem (mainly, Generative Adversarial Networks). 


### Dataset
Dataset used is the public dataset available at [Upslash](https://unsplash.com/data). It provides URL's for 25,000 high definition images. I scraped around 16,500 images in my local storage which is not advisable. You should try storing them at any cloud service (probably google drive) as working with high defintion images cost a lot of resources. 
The code to scrape images is contained in file [data_preparation](https://github.com/kdubelite/Auto-Coloring-of-B-W-images/blob/main/data_preparation.ipynb).

### LAB Color Space
RGB is popular color space in which the image is displayed using three channels: red, blue and green. However, for this problem I have used another color space Lab. LAB[aka CIELAB / L*a*b*] is a colour space that separates brightness and colour entirely. In Lab colour space, a picture has one achromatic luminance(L) channel and two colour channels. The ‘a’ channel is in charge of green and red hues, whereas the ‘b’ channel is in charge of blue and yellow hues. The values for the L channel are typically 0 to 100, while the a and b channels are typically -128 to 127. To get the image, we only need to predict two colour channels, a and b, and then combine them with the L component. [image_analysis](https://github.com/kdubelite/Auto-Coloring-of-B-W-images/blob/main/image_analysis.ipynb)

![alt text](https://github.com/kdubelite/Auto-Coloring-of-B-W-images/blob/main/References/lab_colour_space.png)

Normalization of values is necessary after all the photos have been converted to transform the values in the range of (-1, 1).

### Model Building
The model consists of two parts: the encoder and the decoder. Convolution layers with the activation function 'relu' and strides = 2 are used in the encoder section to reduce the width and height of the latent space vector. The decoder section is made up of convolution layers with upsampling layers to recover the original input image's dimensions (224 × 224) and reconstruct the image with two filters at the last layer that represent the ab channels. In the last layer, 'Tanh' activation is employed to squash the values between -1 and 1. [model](https://github.com/kdubelite/Auto-Coloring-of-B-W-images/blob/main/model.py)

![alt text](https://github.com/kdubelite/Auto-Coloring-of-B-W-images/blob/main/References/model.png)

### Transfer Learning
[VGG19](https://iq.opengenus.org/vgg19-architecture/) model is used in place of encoder. It is trained on ImageNet dataset and is made up of 19 layers. It takes input of 224 x 224 x 3. All images need to be resized to this dimension. As we not doing classification we don't need the classification part of the model. Hence we only build upto the last MaxPooling layer with the same weights as it was originally trained. The model outputs in dimension of 7 x 7 x 512 which serves as input for the decoder.

The model is trained for 500 epochs with optimizer 'adam' and loss function as 'mean square error'.

### Deployment
I have used [streamlit](https://streamlit.io/) web API to create a simple interface to colourize the image. [api](https://github.com/kdubelite/Auto-Coloring-of-B-W-images/blob/main/api.py)
