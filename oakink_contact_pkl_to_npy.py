import os
import pickle
from tqdm import tqdm
import torch
import numpy as np
folder_path = '/home/jihyun/contact_debug/oakink_contact'

oakink_verts_list = []
oakink_contact_list = []

for file_name in tqdm(os.listdir(folder_path)):
    if file_name.endswith('.pkl'):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'rb') as file:
            load_dict = pickle.load(file)

        oakink_verts_list.append(np.expand_dims(load_dict['hand_verts'], axis=0))
        oakink_contact_list.append(np.expand_dims(load_dict['contact'], axis=0))


oakink_verts_npy = np.vstack(oakink_verts_list)
oakink_concat_npy = np.vstack(oakink_contact_list)

print(oakink_verts_npy.shape)
print(oakink_concat_npy.shape)

np.save(f'/home/jihyun/tsne_data/oakink_mano_verts', oakink_verts_npy)
np.save(f'/home/jihyun/tsne_data/oakink_contact', oakink_concat_npy)

