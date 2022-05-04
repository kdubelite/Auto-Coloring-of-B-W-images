# Auto-Coloring-of-B-W-images

In general, colouring a grayscale image is a straightforward process for the human mind; we learn to fill in the mission colours in colouring books from an early age by recalling that the grass is green, the sky is blue, and white clouds are there, or that an apple might be red or green. With the introduction of deep Convolutional neural networks, the challenge has gotten a lot of attention as a representative issue for a thorough visual comprehension of artificial intelligence, akin to what many people previously thought object recognition was.

In this project, I have used the encoder-decoder architecture with concept of transfer learning for auto colourizatoin of black and white images or any image in general. There are various other advanced architectures already in use for this problem (mainly, Generative Adversarial Networks). 


### Dataset
Dataset used is the public dataset available at Upslash. It provides URL's for 25,000 high definition images. I scraped around 16,500 images in my local storage which is not advisable. You should try storing them at any cloud service (probably google drive) as working with high defintion images cost a lot of resources. 
The code to scrape images is contained in file data_preparation.

### LAB Color Space
RGB is popular color space in which the image is displayed using three channels: red, blue and green. However, for this problem I have used another color space Lab. LAB[aka CIELAB / L*a*b*] is a colour space that separates brightness and colour entirely. In Lab colour space, a picture has one achromatic luminance(L) channel and two colour channels. The ‘a’ channel is in charge of green and red hues, whereas the ‘b’ channel is in charge of blue and yellow hues. The values for the L channel are typically 0 to 100, while the a and b channels are typically -128 to 127. To get the image, we only need to predict two colour channels, a and b, and then combine them with the L component. 

Normalization of values is necessary after all the photos have been converted to transform the values in the range of (-1, 1).

### Model Building
The model consists of two parts: the encoder and the decoder. 
