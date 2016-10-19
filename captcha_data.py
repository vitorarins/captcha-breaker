import glob, os
import numpy as np
from scipy import ndimage, misc
import cv2
import random

class OCR_data(object):
    def __init__(self, num, data_dir, num_classes, batch_size=50, len_code=5, height=60, width=180, resize_height=24, resize_width=88, pixel_depth=255.0, num_channels=1):
        self.num = num
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.len_code = len_code
        self.height = height
        self.width = width
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.pixel_depth = pixel_depth
        self.num_channels = num_channels
        self.index_in_epoch = 0
        self._imgs = []
        self._labels = []
        for pathAndFilename in glob.iglob(os.path.join(data_dir, '*.png')):
	    img, label = self.create_captcha(pathAndFilename)
            self._imgs.append(img)
            self._labels.append(label)
        self._imgs = np.array(self._imgs).reshape((-1, resize_height, resize_width, num_channels)).astype(np.float32)
        self._labels = np.array(self._labels)

    def create_captcha(self, pathAndFilename):
        img = cv2.imread(pathAndFilename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(self.resize_width, self.resize_height), interpolation=cv2.INTER_AREA)
        filename, ext = os.path.splitext(os.path.basename(pathAndFilename))
        label = self.create_label(filename)
        return (img, label)

    def create_label(self, filename):
        label = []
        for c in filename:
            ascii_code = ord(c)
            if ascii_code < 58:
                char_value = ascii_code - 48
            else:
                char_value = ascii_code - 87
            label.append(char_value)
        return self.dense_to_one_hot(label, self.num_classes)

    def dense_to_one_hot(self, labels_dense, num_classes):
	num_labels = len(labels_dense)
	index_offest = np.arange(num_labels) * num_classes
	labels_one_hot = np.zeros((num_labels, num_classes))
	labels_one_hot.flat[index_offest + labels_dense] = 1
	labels_one_hot = labels_one_hot.reshape(num_labels*num_classes)
	return labels_one_hot
        
    def next_batch(self, batch_size):
	start = self.index_in_epoch
	self.index_in_epoch += batch_size
	if self.index_in_epoch > self.num:
	    perm = np.arange(self.num)
	    np.random.shuffle(perm)
	    self._imgs = self._imgs[perm]
	    self._labels = self._labels[perm]
	    start = 0
	    self.index_in_epoch = batch_size
	    assert batch_size <= self.num
        end = self.index_in_epoch
        return self._imgs[start:end], self._labels[start:end]
