from tkinter import filedialog as fd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as im


def zero_channels(image: np.ndarray, channel: int):
    a = image.copy()
    for i in range(3):
        if i != channel:
            a[..., i] = 0
    return a


def rgb_decompose(image):
    img = im.open(image)
    a = np.asarray(img).copy()

    red = zero_channels(a, 0)
    green = zero_channels(a, 1)
    blue = zero_channels(a, 2)

    out1 = im.fromarray(red)
    out2 = im.fromarray(green)
    out3 = im.fromarray(blue)

    _, axis = plt.subplots(2, 3)

    axis[0][0].imshow(out1)
    axis[0][0].set_title('Red')

    axis[0][1].imshow(out2)
    axis[0][1].set_title('Green')

    axis[0][2].imshow(out3)
    axis[0][2].set_title('Blue')

    axis[1][0].hist(a[..., 0].ravel(), bins=256, range=(1, 256), color='red')
    axis[1][1].hist(a[..., 1].ravel(), bins=256, range=(1, 256), color='green')
    axis[1][2].hist(a[..., 2].ravel(), bins=256, range=(1, 256), color='blue')

    for i in range(3):
        axis[0][i].axis('off')

    plt.show()
    # out.save('out.jpg')


def main():
    image = fd.askopenfilename()
    rgb_decompose(image)


if __name__ == '__main__':
    main()
