import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
#image_path = "dataset\\train\\train\\00001\\RGB\\uncover\\image_000021.png"

def image_to_array(image_path,crop=True,resize=(40,80)):
    image = Image.open(image_path).convert("L")
    if crop==True:
        image = image.crop((88,100,494,912))
    image = image.resize(resize, Image.Resampling.LANCZOS)
    image_array = np.array(image)
    image_array = 255 - image_array
    image_array = image_array.astype("float32")/255
    return image_array

def load_data(data_path,crop=True,resize=(40,80)):
    image_data = []
    image_label = []
    for class_name in os.listdir(data_path):
        class_dir = os.path.join(data_path, class_name)
        if os.path.isdir(class_dir):
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                image = image_to_array(image_path,crop,resize)
                image_data.append(image)
                image_label.append(int(class_name))
    return np.array(image_data),np.array(image_label)
#image_data,image_label = load_data('dataset\\train_data')
#print(image_label)

def shuffle_data(images,labels):#洗亂data
    num_samples = len(images)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]
    return shuffled_images, shuffled_labels

def plot_image(img):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(img, cmap="binary")
    plt.show()
'''
data,label = load_data('dataset\\train_data')
print(data)
print(label)

'''