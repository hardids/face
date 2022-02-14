
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.express as px
from PIL import Image
import os
import cv2 
#from google.colab.patches import cv2_imshow
import dlib
from skimage import io
import matplotlib.pyplot as plt
from pathlib import Path
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np




def welcome():
    st.title('이 앱은 나의 관상으로 보았을때 어떤 직업이 어울리는지 보는 앱입니다.')
    st.subheader('이미지 또는 캠코더로 직접 입력 해 주세요.')   
    st.image('face_detection.jpeg',use_column_width=True)

def photo():
    st.title('포토파일입력')
    uploaded_file = st.file_uploader("이미지파일선택",type = ["jpg","png","jpeg"])
    
    if uploaded_file is not None:
      image = Image.open(uploaded_file)
      st.image(image, caption='선택된 이미지.', use_column_width=True)
      st.write("")
      st.write("누구일까요")

      # pillow에서 cv로 변환
      numpy_image=np.array(image)  
      opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

      #그레이로 변환
      gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
      st.image(gray, caption='그레이변환.', use_column_width=True)
      # 페이스찾기
      face_detector = dlib.get_frontal_face_detector()
      detected_faces = face_detector(gray, 1)
      face_frames = [(x.left(), x.top(), x.right(), x.bottom()) for x in detected_faces]
      
      for n, face_rect in enumerate(face_frames):
        face = Image.fromarray(opencv_image).crop(face_rect)

      st.image(face, caption='페이스', use_column_width=True)

      # #cv를 pillow로 변환
      # color_coverted = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
      # pil_image=Image.fromarray(color_coverted)

      # st.image(pil_image, caption='PIL페이스', use_column_width=True)


      # Load the model
      model = load_model('keras_model.h5')

      data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

      image = face
      
      # #resize the image to a 224x224 with the same strategy as in TM2:
      # #resizing the image to be at least 224x224 and then cropping from the center
      size = (224, 224)
      image = ImageOps.fit(image, size, Image.ANTIALIAS)

      # #turn the image into a numpy array
      image_array = np.asarray(image)
      # # Normalize the image
      normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
      # # Load the image into the array
      data[0] = normalized_image_array

      # # run the inference

      prediction = model.predict(data)
      st.write(prediction)

      #첫번째 수치를 j에 입력
      label_ = 0
      result1 = "누구일까요?"

     # label_ = prediction[0].index(max(prediction[0]))
      label_ = np.argmax(prediction[0])
          
      if label_ == 0:
          result1 = "정치인" 
      if label_ == 1:
          result1 = "연예인"
      if label_ == 2:
          result1 = "교수"
      if label_ == 3:
          result1 = "CEO"  
      if label_ == 4:
          result1 = "운동선수"

      st.write("나의 최적의 직업은?: "+ result1)


def video():
    st.title('캠코더입력')


selected_box = st.sidebar.selectbox('다음중 선택해주세요',('설명서','사진파일입력', '캠코더입력'))
    
if selected_box == '설명서':
    welcome() 
if selected_box == '사진파일입력':
    photo()
if selected_box == '캠코더입력':
    video()

