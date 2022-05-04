#importing libraries
import streamlit as st
import io
from PIL import Image
from model import cleaning_directories, get_output #importing functions from model.py
from keras.preprocessing.image import load_img
import os

#define the path to input directory
input_path = r'C:\\Users\\kdubelite\\Desktop\\College\\Semester 6\\Data Science\\DS Project\\input'

#setting the tab title
st.set_page_config(
    page_title = 'Image Coloriser',
    page_icon = '😇'
)


st.subheader("Image Colorizer")
image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if image_file:    
    with open(os.path.join(input_path, image_file.name), 'wb') as f:
        f.write((image_file).getbuffer())
    st.success('file saved')
    
    get_output() #getting output

    image_path = r'C:\\Users\\kdubelite\\Desktop\\College\\Semester 6\\Data Science\\DS Project\\output\\out.jpg'
    st.image(load_img(image_path), caption = 'output') #displaying the output

    cleaning_directories() #cleaning the directories

    





