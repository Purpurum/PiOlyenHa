

import numpy as np
from PIL import Image
import os

def crop_image(image, x_center, y_center, width, height):
    left = x_center - width // 2
    top = y_center - height // 2
    right = x_center + width // 2
    bottom = y_center + height // 2
    cropped_image = image.crop((left, top, right, bottom))

    return cropped_image

def process_images(images_path, model):
    unwritten_count = 0
    unwritten_names = []
    path_scan = os.scandir(images_path)
    for image in path_scan:
        try:
            img = Image.open(image.path)
            coords = model.get_animal_coords(image.path)
            print(coords)
            img = crop_image(img, coords[0]["x"], coords[0]["y"], coords[0]["w"], coords[0]["h"])
            img.save("cropped/"+image.name)
            print(image.name + " Written")
        except:
            print("Animel not found on " + image.name)
            unwritten_count+=1
            unwritten_names.append(image.name)
    print("Не опознано на " + unwritten_names)
    print("Не опознано на количестве " + unwritten_count)
        