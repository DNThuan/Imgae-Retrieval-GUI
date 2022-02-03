import pickle

from PIL import Image
import streamlit as st 
from streamlit_cropper import st_cropper

import numpy as np
import os
import time

import ultis
 


st.title("Web retrieval")
st.write("""
Deploying Image Retrieval as Web Application
""")

 
choose_dataset = st.sidebar.selectbox(                            
                            'Dataset',
                            ('oxford5k', 'paris6k'),
                            key=1)
st.write("Dataset: ",choose_dataset)


@st.cache(allow_output_mutation=True)

def load_feature_vectors():
    path_dataset = "VGG_16_4096d_features.npy"
    path = os.path.join(os.getcwd(),'data','dataset', 'oxford5k',path_dataset)

    with open(path,"rb") as f:
        f_data = np.load(f)
    return f_data

def load_kmeans():
    path_kmeans = "Kmean_vgg16_4096d.pkl"
    path = os.path.join(os.getcwd(),'data','dataset','oxford5k',path_kmeans)
    with open(path,"rb") as f:
        kmeans = pickle.load(f)
    return kmeans


features = load_feature_vectors()
kmeans = load_kmeans()

def show_img(img_name):
    list_img = []
    for i in img_name:
        path_image = os.path.join(os.getcwd(),"data/dataset/oxford5k/jpg/{}".format(i))
        try:
            img = Image.open(path_image)
            list_img.append(np.array(img))
        except:
            pass
    return list_img


def save_img(img):
    image = Image.fromarray(img, 'RGB')
    path_save_img = os.path.join(os.getcwd(),"data/image_query/save_qimg.jpg")
    image.save(path_save_img)



uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png'])

realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)

aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
aspect_dict = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "4:3": (4, 3),
    "2:3": (2, 3),
    "Free": None
}
aspect_ratio = aspect_dict[aspect_choice]

if uploadFile:
    img = Image.open(uploadFile)
    rect = st_cropper(
        img,
        realtime_update=realtime_update,

        aspect_ratio=aspect_ratio,
        return_type='box'
    )
    left, top, width, height = tuple(map(int, rect.values()))

    img_croped = np.array(img)[top:top+height, left:left+width]

    save_img(img_croped)
 
    st.write("Preview")
    st.image(img_croped)
    
    if st.button("Search"):

        t = time.time()
        r_ans = ultis.compute_rank(features,kmeans)
        img_list_name = ultis.rank2img(r_ans)
        list_img = show_img(img_list_name)

        st.write("Total time: ", np.around(time.time()-t,2), "s")
        st.image(list_img, width=100)
