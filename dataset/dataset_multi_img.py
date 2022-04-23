import os
import cv2
import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np


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
    # true_folder = folder + 'true'
    # false_folder = folder + 'false'

    # filenames = sorted(os.listdir(true_folder))[:50000]
    # if model_name == 'NPN':
    #     for filename in filenames:
    #         if filename != '.DS_Store':
    #             image_paths.append(os.path.join(true_folder, filename))
    #             labels.append(1)
    #
    #     filenames = sorted(os.listdir(false_folder))[:5000]
    #     for filename in filenames:
    #         if filename != '.DS_Store':
    #             image_paths.append(os.path.join(false_folder, filename))
    #             labels.append(0)
    return image_paths, labels

def load_image_self(path, img_size, stride=32):
    """Load an image using given path.
    """
    img0 = cv2.imread(path)  # BGR
    assert img0 is not None, 'Image Not Found ' + path
    # img0 = img0[:, :, 0]
    img = cv2.resize(img0, (img_size, img_size))



    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
    img = np.ascontiguousarray(img)
    return img