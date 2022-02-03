import imp
import os
import numpy as np
import pandas as pd
from PIL import Image
import Extractquery as EQ




def download_test(data_dir):
    src_dir = 'http://www.robots.ox.ac.uk/~vgg/data/oxbuildings'
    dl_files = ['oxbuild_images.tgz']
  
    dst_dir = os.path.join(data_dir, 'jpg')


    if not os.path.isdir(dst_dir):
   
      os.makedirs(dst_dir)

      for dli in range(len(dl_files)):
          dl_file = dl_files[dli]
          src_file = os.path.join(src_dir, dl_file)

          dst_file = os.path.join(dst_dir, dl_file)
       
          os.system('wget {} -O {}'.format(src_file, dst_file))
 

          # create tmp folder
          dst_dir_tmp = os.path.join(dst_dir, 'tmp')
          os.system('mkdir {}'.format(dst_dir_tmp))
          # extract in tmp folder
          os.system('tar -zxf {} -C {}'.format(dst_file, dst_dir_tmp))
          # remove all (possible) subfolders by moving only files in dst_dir
          os.system('find {} -type f -exec mv -i {{}} {} \\;'.format(dst_dir_tmp, dst_dir))
          # remove tmp folder
          os.system('rm -rf {}'.format(dst_dir_tmp))

          os.system('rm {}'.format(dst_file))




def Sort_Tuple(tup): 
    tup.sort(key = lambda x: x[1]) 
    return tup 

def compute_rank(features,kmeans):
    image_query = "data/image_query/save_qimg.jpg"
    path_image_query = os.path.join(os.getcwd(),image_query)
    img = np.array(Image.open(path_image_query))
    qvec = EQ.extract_feature(img).reshape(1,-1)

    # features = load_feature_vectors()
    # kmeans = load_kmeans()

    label_query = kmeans.predict(qvec.astype(np.float32))

    rank_list_features = []
    rank_list_index = []

    for index,label in enumerate(kmeans.labels_,0):
        if label == label_query:
            rank_list_features.append(features[index])
            rank_list_index.append(index)

    sim = []

    for v in zip(rank_list_features, rank_list_index):
        cos_sim = np.dot(qvec, v[0]) / (np.linalg.norm(qvec) * np.linalg.norm(v[0]))
        sim.append((cos_sim,v[1])) 

    relevant_ans = Sort_Tuple(sim)
    return relevant_ans



def rank2img(relevant_ans):
    path_index_file = "data/dataset/oxford5k/index_new.csv"
    path= os.path.join(os.getcwd(),path_index_file)
    data_df = pd.read_csv(path)
    list_img_name = []
    for ans in relevant_ans[:len(relevant_ans) // 2]:
        list_img_name.append(f"{data_df.iloc[ans[1]][2]}")
    return list_img_name


