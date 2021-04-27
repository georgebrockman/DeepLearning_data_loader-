
import numpy as np
import os
import tensorflow as tf
import random
from PIL import Image
from sklearn.model_selection import train_test_split

class_max = 5000 # change at descretion
batch_size = 256

def dataset_classifcation(path, resize_h, resize_w, train=True, limit=None):
    """ Function to load jpeg images from sub directories for deep learning train purposes """

    # list all paths to data classes except DS_Store
    class_folders = [f for f in sorted(os.listdir(path)) if not f.startswith('.')]
    # load images
    images = []
    classes = []

    train_test_split_num = 0.1
    number_of_test = int(class_max * train_test_split_num)
    for i, c in enumerate(class_folders):
        #images_per_class = sorted(os.path.join(path, c))
        images_per_class = [f for f in sorted(os.listdir(os.path.join(path, c))) if 'jpg' in f]
        # testing inbalanced class theory so limiting to 800 per class - can remove 20-21 later
        if len(images_per_class) > class_max:
            images_per_class = images_per_class[0:class_max]
        #image_class = np.zeros(len(class_folders))
        #image_class[i] = 1

        for image_i, image_per_class in enumerate(images_per_class):
            images.append(os.path.join(path, c, image_per_class))
            classes.append(i)

    train_filenames, val_filenames, train_labels, val_labels = train_test_split(images, classes, train_size=0.9, random_state=420)

    num_train = len(train_filenames)
    num_val = len(val_filenames)

    @tf.function
    def read_images(image_path, class_type, mirrored=False, train=False):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image)

        h, w, c = image.shape
        if not (h == resize_h and w == resize_w):
            image = tf.image.resize(
            image, [resize_h, resize_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # set all images shape to RGB
            image.set_shape((224, 224, 3))

        # data augmentation
        # if train == True:
        #     image = tf.image.random_brightness(image, 0.2)
        #     image = tf.image.random_contrast(image, 0.2, 0.5)
        #     image = tf.image.random_jpeg_quality(image, 75, 100)

        # change DType of image to float32
        image = tf.cast(image, tf.float32)
        class_type = tf.cast(class_type, tf.float32)

        # normalise the image pixels
        image = (image / 255.0)

        return image, class_type
