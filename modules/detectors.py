from ultralytics import YOLO

class detector:
    def __init__(self):
        self.model = YOLO("modules/models/yolov8_detector_nano.pt")

    def get_animal_coords(self, image):
        labels_dict = {
        "0": "number"
        }
        animals = self.model(image)
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

        return preds_list
