import numpy as np
import ml_toolkit as mlt
import pandas as pd
import glob
import time
from PIL import Image


def get_image(path):
    image = Image.open(path).convert('RGB')
    image = image.resize((100, 100))
    image = np.array(image)
    N, D, P = image.shape
    image_vec = image.flatten()
    
    return image_vec, N, D

def form_image(image_vec, N, D, option):
    if option == 'integer':
        image = np.reshape(image_vec, (N, D, 3)).astype(int)
    if option == 'float':
        image = np.reshape(image_vec, (N, D, 3)).astype(float)
    return image
    
def load_covid_images(num_images):
    root_directory = '/Users/Zhonghou/Desktop/General/Classes/'
    root_directory += 'Statistical Modeling II/Project/'
    root_directory += 'COVID-19 Radiography Database/'
    covid_path = root_directory + 'COVID-19/*.png'
    covid_files = glob.glob(covid_path)
    normal_path = root_directory + 'NORMAL/*.png'
    normal_files = glob.glob(normal_path)
    pneumonia_path = root_directory + 'Viral Pneumonia/*.png'
    pneumonia_files = glob.glob(pneumonia_path)
    all_files = [covid_files, normal_files, pneumonia_files]
    class_labels = []
    images = []
    for file_num in range(3):
        n = 0
        files = all_files[file_num]
        for file in files:
            image_vec, N, D = get_image(file)
            images.append(image_vec)
            if file_num == 0:
                class_labels.append(0)
            if file_num == 1:
                class_labels.append(1)
            if file_num == 2:
                class_labels.append(2)
            n += 1
            if n >= num_images:
                break
            

    class_labels = np.array(class_labels)
    images = np.array(images)
    return images, class_labels

def load_cancer_data():
    return
