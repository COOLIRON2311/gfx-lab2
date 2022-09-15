
import tkinter as tk
from tkinter import filedialog as fd

import math
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
        self.canvas = tk.Canvas(self, width=10, height=20, bg='white')
        self.hue = tk.Scale(self, from_=0, to=360, orient=tk.HORIZONTAL, command=self.__hue, label='Hue')
        self.saturation = tk.Scale(self, from_=-50, to=50, orient=tk.HORIZONTAL, command=self.__saturation, label='Saturation')
        self.value = tk.Scale(self, from_=-50, to=50, orient=tk.HORIZONTAL, command=self.__value, label='Value')
        self.canvas.create_text(75, 20, text='No image', anchor=tk.NW)
        self.canvas.config(width=200, height=50)
        self.canvas.pack()
        self.hue.pack(fill=tk.X)
        self.saturation.pack(fill=tk.X)
        self.value.pack(fill=tk.X)
        self.buttons.pack()
        self.open.pack(padx=5, pady=5, side=tk.LEFT)
        self.save.pack(padx=5, pady=5, side=tk.RIGHT)

        self.saturation.set(0)
        self.value.set(0)

    def open_file(self):
        self.filename = fd.askopenfilename()
        img = im.open(self.filename)
        self.data = np.asarray(img).copy()
        self.canvas.config(width=img.width, height=img.height)
        self.title(self.filename)
        # TODO: Set hue, saturation, value to proper values
        self.hsv_data = []
        for i in range(0,len(self.data)):
            self.hsv_data.append([])
            for j in range(0,len(self.data[i])):
                #n_data[i].append([])
                nv = RGBtoHSV(self.data[i][j])
                self.hsv_data[i].append(nv)
                #self.data[i][j] = hlv_data[i][0]
        self.update_image()

    def save_file(self):
        im.fromarray(self.data).save(fd.asksaveasfilename())

    def update_image(self):
        self.image = itk.PhotoImage(im.fromarray(self.data))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

    def __hue(self, hue):
        for i in range(0,len(self.data)):
            for j in range(0,len(self.data[i])):
                h = int(self.hsv_data[i][j][0] + int(hue)) % 360
                s = int(self.hsv_data[i][j][1])
                v = int(self.hsv_data[i][j][2])
                r = HSVtoRGB([h,s,v])
                r[0]*=255
                r[1]*=255
                r[2]*=255
                self.data[i][j] = r
        self.update_image()

    def __saturation(self, saturation):
        sat = int(saturation)
        for i in range(0,len(self.data)):
            for j in range(0,len(self.data[i])):
                h = int(self.hsv_data[i][j][0])
                s = min(abs(int(self.hsv_data[i][j][1]+sat)),100)
                v = int(self.hsv_data[i][j][2])
                r = HSVtoRGB([h,s,v])
                r[0]*=255
                r[1]*=255
                r[2]*=255
                self.data[i][j] = r
        self.update_image()

    def __value(self, value):
        for i in range(0,len(self.data)):
            for j in range(0,len(self.data[i])):
                h = int(self.hsv_data[i][j][0])
                s = int(self.hsv_data[i][j][1])
                v = min(int(abs(self.hsv_data[i][j][2]+int(value))),100)
                r = HSVtoRGB([h,s,v])
                r[0]*=255
                r[1]*=255
                r[2]*=255
                self.data[i][j] = r
        self.update_image()

def HSVtoRGB(pixel):
    H = pixel[0]
    S = pixel[1]/100
    V = pixel[2]/100

    f = H/60 - math.floor(H/60)
    p = V*(1-S)
    q = V*(1-f*S)
    t = V*(1-(1-f)*S)

    Hi = math.floor(H/60)%6

    res = []

    if Hi == 0:
         res = [V,t,p]
    elif Hi == 1:
        res =  [q,V,p]
    elif Hi == 2:
        res= [p,V,t]
    elif Hi == 3:
        res=[p,q,V]
    elif Hi == 4:
        res=[t,p,V]
    elif Hi == 5:
        res= [V,p,q]

    return res

def RGBtoHSV(pixel):
    p = norm_pixel(pixel)
    return [calc_hue(p),calc_satur(p)*100,calc_val(p)*100]

def norm_pixel(pixel):
    s = sum(pixel)
    if s != 0:
        return pixel/255
    else:
        return pixel

def calc_hue(pixel):
    mx = max(pixel)
    mn = min(pixel)

    if mx == mn:
        return 0
    elif mx == pixel[0] and pixel[1]>=pixel[2]:
        return 60*(pixel[1]-pixel[2])/(mx-mn)
    elif mx == pixel[0] and pixel[1]<pixel[2]:
        return 60*(pixel[1]-pixel[2])/(mx-mn)+360
    elif mx == pixel[1]:
        return 60*(pixel[2]-pixel[0])/(mx-mn)+120
    elif mx == pixel[2]:
        return 60*(pixel[0]-pixel[1])/(mx-mn)+240

    return 0

def calc_val(pixel):
    return max(pixel)

def calc_satur(pixel):
    mx = max(pixel)
    mn = min(pixel)

    if max == 0:
        return 0
    else:
        return 1-mn/mx


if __name__ == '__main__':
    Window()
