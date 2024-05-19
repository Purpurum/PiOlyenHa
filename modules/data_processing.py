import numpy as np
from PIL import Image
import os
import time
import csv
import shutil
import tempfile

def form_csv(data):
    fieldnames = ['img_name', 'class']
    print(data)
    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for key, value in data.items():
            writer.writerow({'img_name': key, 'class': value})

def crop_image(image, x_center, y_center, width, height):
    left = x_center - width // 2
    top = y_center - height // 2
    right = x_center + width // 2
    bottom = y_center + height // 2
    cropped_image = image.crop((left, top, right, bottom))

    return cropped_image

def process_images(images_path, model, classifier):
    path_scan = os.scandir(images_path)
    data = {}
    for image in path_scan:
        try:
            start_time = time.time()
            img = Image.open(image.path)
            coords = model.get_animal_coords(image.path)
            print(coords)
            img = crop_image(img, coords[0]["x"], coords[0]["y"], coords[0]["w"], coords[0]["h"])
            result = classifier.inference(image=img)
            #img.save("cropped/"+image.name)
            end_time = time.time()
            print(f"inference full time: {end_time - start_time} seconds")
            print("final_result: " + result )
            print(result)
            data[image.name] = result
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            print("Animel not found on " + image.name)
    form_csv(data)
            
def process_images_zip(images_path, model, classifier):
    ext = images_path.split(".")[-1]
    if ext == ".zip":
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(images_path, temp_dir)
            
            images_path = temp_dir
            
            path_scan = os.scandir(images_path)
            data = {}
            for image in path_scan:
                try:
                    start_time = time.time()
                    img = Image.open(image.path)
                    coords = model.get_animal_coords(image.path)
                    print(coords)
                    img = crop_image(img, coords[0]["x"], coords[0]["y"], coords[0]["w"], coords[0]["h"])
                    result = classifier.inference(image=img)
                    #img.save("cropped/"+image.name)
                    end_time = time.time()
                    print(f"inference full time: {end_time - start_time} seconds")
                    print("final_result: " + result )
                    print(result)
                    data[image.name] = result
                except Exception as e:
                    print(f"An unexpected error occurred: {str(e)}")
                    print("Animal not found on " + image.name)
            form_csv(data)  