from PIL import Image
from PIL import ImageFile
import torchvision.transforms.functional as T
from torchvision import transforms
import os
from joblib import Parallel, delayed
import numpy

def open_image(path):
    return Image.open(path).convert("RGB")

def processing(im):
    img = open_image(im)
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=[0.5,0.5], contrast=[0.5,0.5], saturation=0.1),
        transforms.ColorJitter(hue=0.5),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensor()
    ])

    #new_img = T.adjust_brightness(img, 0.5)
    #new_img = transform(img)
    arr = img * numpy.array([0.1, 0.2, 1.5])
    arr = (120*arr/arr.max()).astype(numpy.uint8)
    new_img = Image.fromarray(arr)
    #new_img = transforms.ToPILImage()(new_img)
    name = im.split("\\")[-1][:-4]
    new_img.save(f"{out}/{name}DA.jpg")


if __name__ == "__main__":
    # Path is fixed, this will not work on other machines
    path_in  = "D:\Prova\CosPlace/ImmaginiProvaDA"
    path_dataset = "D:\Prova\small/train"
    out = "D:\Prova\CosPlace\out"
    images = []

    files = [os.path.join(path, name) for path, subdirs, files in os.walk(path_dataset) for name in files]

    
    results = Parallel(n_jobs=8)(delayed(processing)(im) for im in files)    

