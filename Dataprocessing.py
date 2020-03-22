

import random
import os
import cv2
import numpy as np
from Utils import extract_ids
from pycocotools.coco import COCO
import skimage

import matplotlib.pyplot as plt

def create_dataset(dataset, batch, dict_dir):

    '''
    if dataset == 'Train':
        img_path = os.getcwd() + '/raw/train/images/'
        mask_path = os.getcwd() + '/meta/mask/Train/'
        image_list = random.choices(os.listdir(mask_path), k=batch)

    if dataset == 'Val':
        img_path = os.getcwd() + "/raw/val/images/"
        mask_path = os.getcwd() + "/meta/mask/Val/"
        image_list = random.choices(os.listdir(mask_path), k=batch)
    '''

    img_path = dict_dir[dataset]['Image_Path']
    mask_path = dict_dir[dataset]['Mask_Path']

    image_list = random.choices([id_ for id_ in os.listdir(mask_path) if id_ in os.listdir(img_path)], k=batch)

    img_arr = np.zeros([batch, 128, 128, 3], dtype=np.float32)
    mask_arr = np.zeros([batch, 128, 128], dtype=np.float32)

    for counter, img_file in enumerate(image_list):

        # create Imagedata
        img_ = os.path.normpath(img_path + img_file)
        img = cv2.imread(img_)
        img = img.astype(np.float32)
        img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        # img = np.moveaxis(img, 2, 0)
        img_arr[counter] = img / 255.

        # create Mask

        mask = cv2.imread(os.path.normpath(mask_path + img_file), cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32)
        mask = cv2.resize(mask, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        mask_arr[counter] = mask / 255.

    return img_arr, np.expand_dims(mask_arr, axis=3)


def generator(Dataset, batch, dict_dir):
    # Create empty arrays to contain batch of features and labels#

    while True:

        train_img, train_mask = create_dataset(Dataset, batch, dict_dir)

        if random.randint(0, 7) == 1:
            print('--- augmented batch called ----')

            train_img_aug = np.zeros([batch, 128, 128, 3], dtype=np.float32)
            train_mask_aug = np.zeros([batch, 128, 128], dtype=np.float32)

            for counter, _ in enumerate(train_img):

                choice = random.choice(['flip_h', 'flip_v'])

                if choice == 'flip_v':

                    train_img_aug[counter] = np.fliplr(train_img[counter, :, :, :])
                    train_mask_aug[counter] = cv2.resize(np.fliplr(train_mask[counter, :, :, :]), dsize=(128, 128))

                if choice == 'flip_h':
                    train_img_aug[counter] = np.flipud(train_img[counter, :, :, :])
                    train_mask_aug[counter] = cv2.resize(np.fliplr(train_mask[counter, :, :, :]), dsize=(128, 128))

                #if choice == 'noise':
                #    train_img_aug[counter] = skimage.util.random_noise(train_img[counter, :, :, :], mode='gaussian')
                #    train_mask_aug[counter] = cv2.resize(train_mask[counter], dsize=(128, 128))

            yield train_img_aug, np.expand_dims(train_mask_aug, axis=3)

        else:
            yield train_img, train_mask


def create_mask(dataset, batch_mask, dict_dir):

    mask_exists = [extract_ids(f) for f in os.listdir(dict_dir[dataset]['Mask_Path'])]
    global coco

    coco = COCO(dict_dir[dataset]['Ann_Small'])

    image_ids = [id_ for id_ in load_ann_ids(dataset, dict_dir) if id_ not in mask_exists]

    for img_ in image_ids[0:batch_mask]:

        annotations = annotation_image(img_)
        mask = coco.annToMask(annotations[0])

        for i in range(len(annotations)):
            mask += coco.annToMask(annotations[i])

        mask = mask.reshape((300, 300))
        file_name = str(img_).zfill(12) + str('.jpg')
        print('File created:', file_name)

        mask[mask != 0] = 1

        plt.viridis()
        plt.axis('off')
        plt.imshow(mask)
        plt.savefig(dict_dir[dataset]['Mask_Path'] + file_name, bbox_inches='tight', pad_inches=0)


def annotation_image(id_):
    annotation_ids = coco.getAnnIds(imgIds=id_)
    annotations = coco.loadAnns(annotation_ids)

    return annotations


def load_ann_ids(dataset, dict_dir):

    coco = COCO(dict_dir[dataset]['Ann_Small'])
    category_ids = coco.loadCats(coco.getCatIds())
    image_ids = coco.getImgIds(catIds=coco.getCatIds())

    return image_ids


