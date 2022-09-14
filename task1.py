from tkinter import filedialog as fd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as im


def image_2_grayscale(image):
    img = im.open(image)
    a = np.asarray(img).copy()
    ntsc = np.dot(a[..., :3], [0.299, 0.587, 0.114])
    hdtv = np.dot(a[..., :3], [0.2126, 0.7152, 0.0722])

    gs_diff = abs(hdtv - ntsc)

    ntsc = ntsc.astype(np.uint8)
    hdtv = hdtv.astype(np.uint8)
    gs_diff = gs_diff.astype(np.uint8)

    out1 = im.fromarray(ntsc)
    out2 = im.fromarray(hdtv)
    out_diff = im.fromarray(gs_diff)

    _, axis = plt.subplots(2, 3)
    axis[0][0].imshow(img)
    axis[0][0].set_title('Original')

    axis[0][1].imshow(out1, cmap='gray')
    axis[0][1].set_title('PAL/NTSC')

    axis[0][2].imshow(out2, cmap='gray')
    axis[0][2].set_title('HDTV')

    axis[1][0].imshow(out_diff, cmap='gray')
    axis[1][0].set_title('Difference')

    axis[1][1].plot(range(0, 256), np.histogram(ntsc, bins=256)[0])
    axis[1][2].plot(range(0, 256), np.histogram(hdtv, bins=256)[0])

    for i in range(2):
        for j in range(3):
            if i == 1 and j > 0:
                continue
            axis[i][j].axis('off')

    plt.show()
    # out.save('out.jpg')


def main():
    image = fd.askopenfilename()
    image_2_grayscale(image)


if __name__ == '__main__':
    main()
