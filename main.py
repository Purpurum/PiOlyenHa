from modules.detectors import detector, classifierResnetF
from modules.data_processing import process_images

import tkinter as tk
from tkinter import filedialog

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.core.window import Window

import os

CHOOSE = False

detector = detector()
classifier = classifierResnetF()

class AnimalLocatorApp(App):
    def build(self):
        #returns a window object with all it's widgets
        self.window = GridLayout()
        self.window.cols = 1
        self.window.size_hint = (0.6, 0.7)
        self.window.pos_hint = {"center_x": 0.5, "center_y":0.5}

        # image widget
        self.img = Image(source="")
        Window.bind(on_drop_file=self.on_drop_file)
        self.window.add_widget(self.img)

        # label widget
        self.greeting = Label(
                        text= "Впишите путь к папке",
                        font_size= 18,
                        color= '#00FFCE'
                        )
        self.window.add_widget(self.greeting)

        # text input widget
        self.user = TextInput(
                    multiline= False,
                    padding_y= (20,20),
                    size_hint= (1, 0.5)
                    )

        self.window.add_widget(self.user)

        # button widget
        self.button = Button(
                      text= "Выбрать папку по пути",
                      size_hint= (1,0.5),
                      bold= True,
                      background_color ='#00FFCE',
                      )
        self.window.add_widget(self.button)

        # button widget
        self.button1 = Button(
                      text= "Выбрать папку",
                      size_hint= (1,0.5),
                      bold= True,
                      background_color ='#00FFCE',
                      )
        self.button1.bind(on_press=self.choose_folder)
        self.window.add_widget(self.button1)

        return self.window
    
    def choose_folder(self, instance):
        root = tk.Tk()
        root.withdraw()
        pathVar = filedialog.askdirectory()
        process_images(pathVar, detector, classifier)
    
    def on_drop_file(self, window, file_path_bytes, x, y):
        decoded_str = file_path_bytes.decode('utf-8')
        file_path = os.path.abspath(decoded_str)
        print(file_path)
        self.img.source=file_path
        
        process_images(file_path, detector, classifier)
        
        return

if __name__ == "__main__":
    AnimalLocatorApp().run()