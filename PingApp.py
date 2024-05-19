from modules.detectors import detector, classifierResnetF, classifierEnseble
from modules.data_processing import process_images, process_images_zip



import threading
from queue import Queue

import tkinter as tk
from tkinter import filedialog

from kivy.config import Config
Config.set('graphics', 'width', '1000')
Config.set('graphics', 'height', '600')
Config.set("graphics","resizable", '0')
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.properties import ListProperty, StringProperty, NumericProperty, ObjectProperty
from kivy.graphics import Color, Ellipse

USE_ENSEMBLE = False

thread_complete = threading.Event()

results = {}


detector = detector()
if USE_ENSEMBLE:
    classifier = classifierEnseble()
else:
    classifier = classifierResnetF()

def process_images_thread(file_queue, detector, classifier):
        while True:
            try:
                pathVar = file_queue.get(block=False)
                process_images(pathVar, detector, classifier)
                file_queue.task_done()
            except:
                break
        thread_complete.set()

def process_images_thread_zip(file_queue, detector, classifier):
        while True:
            try:
                pathVar = file_queue.get(block=False)
                process_images_zip(pathVar, detector, classifier)
                file_queue.task_done()
            except:
                break
        thread_complete.set()       

class PingLayout(Widget):
    status_text = StringProperty()
    file_queue = ObjectProperty(Queue())

    def press_openfile(self):
        root = tk.Tk()
        root.withdraw()
        pathVar = filedialog.askdirectory()
        self.file_queue.put(pathVar)
        threading.Thread(target=process_images_thread, args=(self.file_queue, detector, classifier)).start()
    
    def change_image(self):
        self.ids.my_image.source = 'plot.png'

class PingApp(App):
    file_queue = ObjectProperty(Queue())
    def build(self):
        self.layout = PingLayout()
        self.color = Color(rgba=(0.1, 0.9, 0.1, 1))
        self.layout.canvas.add(self.color)
        self.layout.canvas.add(Ellipse(pos=self.layout.pos, size=(20, 20)))
        Window.bind(on_drop_file=self.on_drop_file)
        Window.clearcolor = (1, 1, 1, 1)
        return self.layout
    
    def on_drop_file(self, window, file_path_bytes, x, y):
        file_path = file_path_bytes.decode('utf-8')
        self.layout.status_text = file_path
        self.file_queue.put(file_path)
        self.color.rgba = (0.9, 0.1, 0.1, 1)
        if file_path.split(".")[-1] != ".zip":
            threading.Thread(target=process_images_thread, args=(self.file_queue, detector, classifier)).start()
        else:
            threading.Thread(target=process_images_thread_zip, args=(self.file_queue, detector, classifier)).start()

        try:
            def wait_for_thread_to_finish():
                thread_complete.wait()
                self.color.rgba = (0.1, 0.9, 0.1, 1)

            wait_thread = threading.Thread(target=wait_for_thread_to_finish)
            wait_thread.start()
        except:
            self.color.rgba = (0.1, 0.9, 0.1, 1)

if __name__ == "__main__":
    PingApp().run()