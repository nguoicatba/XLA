import tkinter.messagebox
from tkinter import *;
from tkinter import filedialog
import tkinter as tk
import os
import cv2
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import  Model
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np



def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs = vgg16_model.get_layer("fc2").output)
    return extract_model


def image_preprocess(img):
    img = img.resize((224,224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_vector(model, image_path):
    print("Xu ly : ", image_path)
    img = Image.open(image_path)
    img_tensor = image_preprocess(img)


    vector = model.predict(img_tensor)[0]

    vector = vector / np.linalg.norm(vector)
    return vector

model1=get_extract_model()
def clustering_folder_image(k,path):
    input_folder=path
    vectors = []
    k=int(k)
    image_path = []
    image_name = []
    for image1 in os.listdir(input_folder):
        path_image = os.path.join(input_folder, image1);
        vector_dactrung = extract_vector(model1, path_image);
        vectors.append(vector_dactrung)
        image_path.append(path_image)
        image_name.append(image1)

    vector1 = np.array(vectors);
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(vector1)
    labels = kmeans.labels_;
    centroid = kmeans.cluster_centers_
    print(kmeans.labels_)
    print(centroid)
    index = 0;
    for label_image in labels:
        path_output = input_folder + "\\" + "cluster" + str(label_image);
        if (os.path.isdir(path_output) == False):
            os.makedirs(path_output);

        make_image = cv2.imread(image_path[index])
        cv2.imwrite(path_output + "\\" + image_name[index], make_image)
        index = index + 1;
        
    for path in image_path:
        os.remove(path)


def choice():
    inp = text1.get(1.0, "end-1c")
    if inp.isnumeric():
        var = filedialog.askdirectory()
        if var == "":
            return;
        clustering_folder_image(inp,var)
        tkinter.messagebox.showinfo("Phân loại ảnh", "Phân loại thành công")
    else:
        return;


root = Tk()
root.geometry("200x200")
root.title("Phân loại ảnh")
label = Label(root, text="Phân loại ảnh trong thư mục")
label.place(x=30, y=60)
label.pack
text1 = tk.Text(root, height=1, width=15)
text1.place(x=30, y=100)
text1.pack;
button = Button(root, text="Phân loại ảnh", command=choice).place(x=30, y=150)
lbl = tk.Label(root, text="")
lbl.pack()
root.mainloop()




