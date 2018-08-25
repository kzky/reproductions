from nnabla.utils.data_iterator import data_iterator_cache, data_iterator_simple
from PIL import Image
import glob
import numpy as np


# def data_iterator_imagenet(batch_size, cache_dir, rng=None):
#     return data_iterator_cache(cache_dir, batch_size, shuffle=True, normalize=False, rng=rng)


def data_iterator_imagenet(img_path, dirname_to_label_path,
                           batch_size=16, ih=128, iw=128,
                           train=True, shuffle=True, rng=None):
    # Images
    imgs = glob.glob("{}/*/*.JPEG".format(img_path))

    # Dirname to Label map
    dirname_to_label = {}
    label_to_dirname = {}
    with open(dirname_to_label_path) as fp:
        for l in fp:
            d, l = l.rstrip().split(" ")
            dirname_to_label[d] = int(l)
            label_to_dirname[int(l)] = d


    def load_func(i):
        # image
        img = Image.open(imgs[i]).resize((iw, ih), Image.BILINEAR)
        img = np.asarray(img)
        if len(img.shape) != 3:
            img = img.reshape((ih, iw, 1))
            img = np.concatenate([img, img, img], axis=2)
        img = img.transpose((2, 0, 1))
        img = img / 128.0 - 1.0
        img += np.random.uniform(size=img.shape, low=0.0, high=1.0 / 128)
        # label
        elms = imgs[i].rstrip().split("/")
        dname = elms[-2]
        label = dirname_to_label[dname]
        lable = np.asarray(label)
        return img, label
        

    return data_iterator_simple(
        load_func, len(imgs), batch_size, shuffle=shuffle, rng=rng, with_file_cache=False)


def main():
    img_path = "/home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/train_cache_sngan"
    dirname_to_label_path = "/home/Kazuki.Yoshiyama/data/imagenet/sngan_projection/dirname_to_label.txt"
    di = data_iterator_imagenet(img_path, dirname_to_label_path)
    itr = 1620
    for i in range(itr):
        x, y = di.next()
        print(i, x.shape)
        print(i, y.shape)
        if x.shape != (16, 3, 128, 128):
            break

if __name__ == '__main__':
    main()
