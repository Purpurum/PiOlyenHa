from ultralytics import YOLO
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import time

class classifierEnseble:
    def __init__(self): 
        device = torch.device("cpu")
        self.modelM = models.resnext50_32x4d()
        self.modelM.load_state_dict(torch.load('modules/models/resnext50_32x4d_classifier240.pth'))
        self.modelM.eval()
        self.modelO = models.resnext50_32x4d()
        self.modelO.load_state_dict(torch.load('modules/models/best_model-ol.pth'))
        self.modelO.eval()
        self.modelKab = models.resnext50_32x4d()
        self.modelKab.load_state_dict(torch.load('modules/models/best_model-kab.pth'))
        self.modelKab.eval()
        self.modelKos = models.resnext50_32x4d()
        self.modelKos.load_state_dict(torch.load('modules/models/best_model-kos.pth'))
        self.modelKos.eval()
    
    def inferenceM(self, image):
        start_time = time.time()
        transform_pipeline = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform_pipeline(image)
        image = image.unsqueeze(0)
        
        output = self.modelM(image)
        _, predicted_class_index = torch.max(output, dim=1)
        end_time = time.time()
        print(f"inference classM time: {end_time - start_time} seconds")
        return predicted_class_index.item()
    
    def inferenceO(self, image):
        start_time = time.time()
        transform_pipeline = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform_pipeline(image)
        image = image.unsqueeze(0)
        
        output = self.modelO(image)
        _, predicted_class_index = torch.max(output, dim=1)
        end_time = time.time()
        print(f"inference classM time: {end_time - start_time} seconds")
        return predicted_class_index.item()
    
    def inferenceKos(self, image):
        start_time = time.time()
        transform_pipeline = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform_pipeline(image)
        image = image.unsqueeze(0)
        
        output = self.modelKos(image)
        _, predicted_class_index = torch.max(output, dim=1)
        end_time = time.time()
        print(f"inference classM time: {end_time - start_time} seconds")
        return predicted_class_index.item()
    
    def inferenceKab(self, image):
        start_time = time.time()
        transform_pipeline = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform_pipeline(image)
        image = image.unsqueeze(0)
        
        output = self.modelKab(image)
        _, predicted_class_index = torch.max(output, dim=1)
        end_time = time.time()
        print(f"inference classM time: {end_time - start_time} seconds")
        return predicted_class_index.item()
    
    def weighted_sum(self, results):
        def find_max_in_dict(my_dict):
            max_key = None
            max_value = float('-inf')  # Initialize with negative infinity
            
            for key, value in my_dict.items():
                if value > max_value:
                    max_value = value
                    max_key = key
            
            return max_key, max_value
        print(results)
        counters = {
            "kab_counter" : 0,
            "ol_counter" : 0,
            "kos_counter" : 0,
        }
        
        if results[0] == 0: counters["kab_counter"]+=10
        elif results[0] == 1: counters["kos_counter"]+=10
        elif results[0] == 2: counters["ol_counter"]+=10

        if results[1] == 0: counters["ol_counter"]-=5
        elif results[1] != 0: counters["ol_counter"]+=5

        if results[2] == 0: counters["kos_counter"]-=5
        elif results[2] != 0: counters["kos_counter"]+=5

        if results[3] == 0: counters["kab_counter"]-=5
        elif results[3] != 0: counters["kab_counter"]+=5

        result = find_max_in_dict(counters)
        print(counters)
        print(result)
        if result[0] == "kab_counter":
            #return "Кабарга"
            return "0"
        elif result[0] == "kos_counter":
            #return "Косуля"
            return "1"
        elif result[0] == "ol_counter":
            #return "Олень"
            return "2"

    def inference(self, image):
        resM = self.inferenceM(image)
        resO = self.inferenceO(image)
        resKos = self.inferenceKos(image)
        resKab = self.inferenceKab(image)
        results = [resM, resO, resKos, resKab]
        final_result = self.weighted_sum(results)
        return final_result
        
class classifierResnetF:
    def __init__(self): 
        device = torch.device("cpu")
        self.model = models.resnext50_32x4d()
        self.model.load_state_dict(torch.load('modules/models/resnext50_32x4d_classifier240.pth'))
        self.model.eval()

    def inference(self, image):
        start_time = time.time()
        transform_pipeline = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform_pipeline(image)
        image = image.unsqueeze(0)
        
        output = self.model(image)
        _, predicted_class_index = torch.max(output, dim=1)
        end_time = time.time()
        print(f"inference class time: {end_time - start_time} seconds")
        if predicted_class_index.item() == 0:
            #return "Кабарга"
            return "0"
        elif predicted_class_index.item() == 1:
            #return "Косуля"
            return "1"
        elif predicted_class_index.item() == 2:
            #return "Олень"
            return "2"


class detector:
    def __init__(self):
        self.model = YOLO("modules/models/yolov8_detector_nano.pt")

    def get_animal_coords(self, image):
        start_time = time.time()
        animals = self.model(image, device="CPU")
        preds_list = []
        for animal in animals:
            prediction = {
            'x': int(animal.to("cpu").numpy().boxes.xywh[:, 0][0]),
            'y': int(animal.to("cpu").numpy().boxes.xywh[:, 1][0]),
            'w': int(animal.to("cpu").numpy().boxes.xywh[:, 2][0]),
            'h': int(animal.to("cpu").numpy().boxes.xywh[:, 3][0]),
            }
            
            #prediction['confidence'] = str(animal.to("cpu").numpy().boxes.conf[0])
            #prediction['class'] = str((animal.to("cpu").numpy().boxes.cls)[0].astype(int))
            #prediction['name'] = prediction["class"].replace(prediction['class'], labels_dict[prediction['class']])
            preds_list.append(prediction)
        end_time = time.time()
        print(f"inference detect time: {end_time - start_time} seconds")
        return preds_list
