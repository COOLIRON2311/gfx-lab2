import tkinter as tk
from tkinter import filedialog as fd

import numpy as np
from PIL import Image as im
from PIL import ImageTk as itk


class Window(tk.Tk):
    filename: str
    data: np.ndarray
    image: 'tk._CanvasItemId'

    def __init__(self):
        super().__init__()
        self.resizable(False, False)
        self.create_widgets()
        self.mainloop()

    def create_widgets(self):
        self.buttons = tk.Frame(self)
        self.open = tk.Button(self.buttons, text='Open', command=self.open_file)
        self.save = tk.Button(self.buttons, text='Save', command=self.save_file)
        self.canvas = tk.Canvas(self, width=10, height=20)
        self.hue = tk.Scale(self, from_=0, to=360, orient=tk.HORIZONTAL, command=self.__hue, label='Hue')
        self.saturation = tk.Scale(self, from_=0, to=100, orient=tk.HORIZONTAL, command=self.__saturation, label='Saturation')
        self.value = tk.Scale(self, from_=0, to=100, orient=tk.HORIZONTAL, command=self.__value, label='Value')
        self.canvas.pack()
        self.hue.pack(fill=tk.X)
        self.saturation.pack(fill=tk.X)
        self.value.pack(fill=tk.X)
        self.buttons.pack()
        self.open.pack(padx=5, pady=5, side=tk.LEFT)
        self.save.pack(padx=5, pady=5, side=tk.RIGHT)

    def open_file(self):
        self.filename = fd.askopenfilename()
        img = im.open(self.filename)
        self.data = np.asarray(img).copy()
        self.canvas.config(width=img.width, height=img.height)
        self.title(self.filename)
        self.update_image()

    def save_file(self):
        im.fromarray(self.data).save(fd.asksaveasfilename())

    def update_image(self):
        self.image = itk.PhotoImage(im.fromarray(self.data))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

    def __hue(self, hue):
        ...
        self.update_image()

    def __saturation(self, saturation):
        ...
        self.update_image()

    def __value(self, value):
        ...
        self.update_image()


if __name__ == '__main__':
    Window()
