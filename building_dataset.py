import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


images_url = '../../data/leftImg8bit_trainvaltest/leftImg8bit/train'
images_list = glob.glob(os.path.join(images_url, '*_leftImg8bit.png'))
lables_url = '../../data/gtFine_trainvaltest/gtFine/train'
lables_list = glob.glob(os.path.join(lables_url, "*_labelIds.png"))


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def crop(image, indexes):
    x = indexes[0][0]
    y = indexes[1][0]
    print(x, y)
    crop_image = image[x:x + 82, y:y + 82]
    plt.imshow(crop_image)
    plt.show(block=True)


img = np.array(Image.open(lables_list[0]))
index_of_tfl = np.where(img == 19)
index_of_not_tfl = np.where(img != 19)

original_img = np.array(Image.open(images_list[0]))

big_original_img = np.pad(original_img, 40, pad_with)[:, :, 40:43]

crop(big_original_img, index_of_tfl)
crop(big_original_img, index_of_not_tfl)
