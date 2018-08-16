import numpy as np
from PIL import Image
import cv2
import os


def psnr(x, y, max_=255):
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mse = np.mean((x - y) ** 2)
    return 10 * np.log10(max_ ** 2 / mse)


def main():
    home = os.path.expanduser("~")
    img_path = "{}/nnabla_data/Set14/monarch.png".format(home)

    # PIL
    img = Image.open(img_path)
    w, h = img.size
    sw, sh = w // 4, h // 4
    img_r = img.resize((sw, sh), Image.BICUBIC)\
                     .resize((w, h), Image.BICUBIC)
    img_array = np.asarray(img)
    img_r_array = np.asarray(img_r)
    print("PSNR (PIL) = {}".format(psnr(img_array, img_r_array)))

    # OpenCV
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    sh, sw = h // 4, w // 4
    img_r = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_CUBIC)
    img_r = cv2.resize(img_r, (w, h), interpolation=cv2.INTER_CUBIC)
    print("PSNR (CV2) = {}".format(psnr(img, img_r)))

    # Matlab (python-impl)
    #TODO: 

if __name__ == '__main__':
    main()
