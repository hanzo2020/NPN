import os
import cv2
import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageChops


class DatasetMImg(torch.utils.data.Dataset):
    def __init__(self, dataset, split, model_name, img_size, class_num):
        self.img_size = img_size
        self.model_name = model_name
        self.class_num = class_num
        assert split in {
            "train",
            "val",
            "test",
        }
        self.image_paths, self.labels = load_images_and_labels(
            dataset=dataset, split=split, model_name=self.model_name, class_num = self.class_num
        )

    def __getitem__(self, item):
        image = load_image_self(
            self.image_paths[item], img_size=self.img_size)
        image = torch.from_numpy(image).type(torch.float32)

        label = torch.tensor(self.labels[item], dtype=torch.float32)
        return image, label

    def __len__(self):
        return len(self.labels)


def expand2square(img, background_color=255):
    size = img.shape
    height = size[0]
    width = size[1]
    # width, height = img.size
    if width == height:
        return img
    elif width > height:
        i = (width - height) // 2
        result = cv2.copyMakeBorder(img, i, i, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        return result
    else:
        i = (height - width) // 2
        result = cv2.copyMakeBorder(img, 0, 0, i, i, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        return result

def load_images_and_labels(dataset='shape', split='train', model_name='NPN', class_num = 3, img_size=128):
    image_paths = []
    labels = []
    folder = 'data/' + dataset + '/' + split + '/'
    for i in range(class_num):
        folder_name = folder + str(i)
        filenames = sorted(os.listdir(folder_name))[:50000]
        for filename in filenames:
            if filename != '.DS_Store':
                image_paths.append(os.path.join(folder_name, filename))
                labels.append(i)
    return image_paths, labels

def load_image_self(path, img_size, stride=32):
    """Load an image using given path.
    """
    img0 = cv2.imread(path)  # BGR
    # img0 = Image.open(path)
    assert img0 is not None, 'Image Not Found ' + path
    # img0 = img0[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
    img0 = expand2square(img0)
    # img = img0.resize((img_size, img_size),Image.ANTIALIAS)
    img = cv2.resize(img0, (img_size, img_size))



    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
    img = np.ascontiguousarray(img)
    return img