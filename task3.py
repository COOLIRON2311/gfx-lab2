import tkinter as tk
from math import floor
from tkinter import filedialog as fd

import numpy as np
from numba import njit, prange
from PIL import Image as im
from PIL import ImageTk as itk


class Window(tk.Tk):
    filename: str
    data: np.ndarray
    image: 'tk._CanvasItemId'
    hsv_data: np.ndarray

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
        self.saturation = tk.Scale(self, from_=-50, to=50, orient=tk.HORIZONTAL,
                                   command=self.__saturation, label='Saturation')
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
        self.hsv_data = np.zeros((img.height, img.width, 3))
        for i in range(0, self.hsv_data.shape[0]):
            for j in range(0, self.hsv_data.shape[1]):
                self.hsv_data[i][j] = RGBtoHSV(self.data[i][j])
        self.update_image()

    def save_file(self):
        im.fromarray(self.data).save(fd.asksaveasfilename())

    def update_image(self):
        self.image = itk.PhotoImage(im.fromarray(self.data))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

    def __hue(self, hue):
        @njit(parallel=True, cache=True)
        def hue_loop(hue, data: np.ndarray, hsv_data):
            for i in prange(0, data.shape[0]):
                for j in prange(0, data.shape[1]):
                    h = int(hsv_data[i][j][0]+hue)
                    s = int(hsv_data[i][j][1])
                    v = int(hsv_data[i][j][2])
                    r = HSVtoRGB([h, s, v])
                    data[i][j] = r
            return data
        self.data = hue_loop(int(hue), self.data, self.hsv_data)
        self.update_image()

    def __saturation(self, saturation):
        @njit(parallel=True, cache=True)
        def saturation_loop(saturation, data: np.ndarray, hsv_data):
            for i in prange(0, data.shape[0]):
                for j in prange(0, data.shape[1]):
                    h = int(hsv_data[i][j][0])
                    s = min(int(abs(hsv_data[i][j][1]+saturation)), 100)
                    v = int(hsv_data[i][j][2])
                    r = HSVtoRGB([h, s, v])
                    data[i][j] = r
            return data
        self.data = saturation_loop(int(saturation), self.data, self.hsv_data)
        self.update_image()

    def __value(self, value):
        @njit(parallel=True, cache=True)
        def value_loop(value, data: np.ndarray, hsv_data):
            for i in prange(0, data.shape[0]):
                for j in prange(0, data.shape[1]):
                    h = int(hsv_data[i][j][0])
                    s = int(hsv_data[i][j][1])
                    v = min(int(abs(hsv_data[i][j][2]+value)), 100)
                    r = HSVtoRGB([h, s, v])
                    data[i][j] = r
            return data
        self.data = value_loop(int(value), self.data, self.hsv_data)
        self.update_image()


@njit(cache=True, inline='always')
def HSVtoRGB(pixel: np.ndarray) -> np.ndarray:
    H = pixel[0]
    S = pixel[1]/100
    V = pixel[2]/100

    f = H/60 - floor(H/60)
    p = V*(1-S)
    q = V*(1-f*S)
    t = V*(1-(1-f)*S)

    Hi = floor(H/60) % 6

    match Hi:
        case 0:
            return np.array([V, t, p]) * 255
        case 1:
            return np.array([q, V, p]) * 255
        case 2:
            return np.array([p, V, t]) * 255
        case 3:
            return np.array([p, q, V]) * 255
        case 4:
            return np.array([t, p, V]) * 255
        case 5:
            return np.array([V, p, q]) * 255


@njit(cache=True, inline='always')
def RGBtoHSV(pixel: np.ndarray) -> np.ndarray:
    p = norm_pixel(pixel)
    return np.array([calc_hue(p), calc_satur(p)*100, calc_val(p)*100])


@njit(cache=True, inline='always')
def norm_pixel(pixel: np.ndarray) -> np.ndarray:
    return pixel / 255


@njit(cache=True, inline='always')
def calc_hue(pixel: np.ndarray) -> np.ndarray:
    if pixel[0] == pixel[1] and pixel[1] == pixel[2]:
        return 0
    if pixel[0] >= pixel[1] and pixel[0] >= pixel[2]:
        return 60 * ((pixel[1] - pixel[2]) / (max(pixel) - min(pixel)))
    elif pixel[1] >= pixel[0] and pixel[1] >= pixel[2]:
        return 60 * ((pixel[2] - pixel[0]) / (max(pixel) - min(pixel))) + 120
    else:
        return 60 * ((pixel[0] - pixel[1]) / (max(pixel) - min(pixel))) + 240


@njit(cache=True, inline='always')
def calc_val(pixel: np.ndarray) -> int:
    return max(pixel)


@njit(cache=True, inline='always')
def calc_satur(pixel: np.ndarray) -> int:
    mx = max(pixel)
    mn = min(pixel)

    if mx == 0:
        return 0
    else:
        return 1-mn/mx


if __name__ == '__main__':
    Window()
