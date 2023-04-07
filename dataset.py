#  Copyright (c) Ikara Vision Systems GmbH, Kaiserslautern - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential

__author__ = "gpillai"

from utils.evaluate import make_weights_for_balanced_classes

"""
module for dataset creation and preprocessing
"""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2
from torch.utils.data import DataLoader, SubsetRandomSampler, sampler
from torchvision.transforms import transforms
import torch


class ClassifierData:
    """
    Class for the dataloader and transformations
    """

    @staticmethod
    def data_loader(train_data, val_data, test_data=None, test_size=None, batch_size=32, balance_dataset=False,
                    display_weighted_classes=False):
        """
        A simple dataloader function used for training
        :param train_data: train ImageFolder
        :param val_data: validation ImageFolder
        :param test_data: test ImageFolder
        :param test_size: size to split into test dataset
        :param batch_size: batch size for training
        :param balance_dataset: Indicate whether to balance the under sampled class in the dataset
        :param display_weighted_classes: Display the number of samples per class after the weighted sampling
        :return: dataloaders
        """

        assert (train_data is not None and val_data is not None)
        assert (test_data is not None or test_size is not None)

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)

        if not balance_dataset:
            print("Creating dataloaders!!")

            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

            # if only test data is none but a test size is present, create a test dataset from the training dataset
            if test_data is None and test_size is not None:
                print("Creating test split from train dataset")
                len_data = len(train_data)
                indices = list(range(len_data))
                np.random.shuffle(indices)
                split_data = int(np.floor(test_size * len_data))
                train_id = indices[split_data:]
                test_id = indices[:split_data]
                test_sample = SubsetRandomSampler(test_id)
                train_sample = SubsetRandomSampler(train_id)
                test_loader = DataLoader(train_data, batch_size=batch_size, sampler=test_sample, drop_last=True)
                valid_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
                train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sample)
                dataloaders = {'train': train_loader, 'val': valid_loader, 'test': test_loader}

            else:
                valid_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
                dataloaders = {'train': train_loader, 'val': valid_loader, 'test': test_loader}
                print("Dataset Loaders Created")

        else:
            print("Creating balanced dataset loaders !!")
            weights_classes = make_weights_for_balanced_classes(train_data.imgs, len(train_data.classes))
            weights_classes = torch.DoubleTensor(weights_classes)

            weighted_sampler = sampler.WeightedRandomSampler(weights_classes, len(weights_classes))
            train_loader_weighted = DataLoader(train_data, sampler=weighted_sampler, batch_size=batch_size)

            weights_classes_val = make_weights_for_balanced_classes(val_data.imgs, len(val_data.classes))
            weights_classes_val = torch.DoubleTensor(weights_classes_val)

            weighted_sampler_val = sampler.WeightedRandomSampler(weights_classes_val, len(weights_classes_val))
            val_loader_weighted = DataLoader(val_data, sampler=weighted_sampler_val, batch_size=batch_size)

            if test_data is None and test_size is not None:
                len_data = len(train_data)
                indices = list(range(len_data))
                np.random.shuffle(indices)
                split_data = int(np.floor(test_size * len_data))
                test_id = indices[split_data:]
                test_sample = SubsetRandomSampler(test_id)
                test_loader = DataLoader(train_data, batch_size=batch_size, sampler=test_sample)
                valid_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
                dataloaders = {'train': train_loader_weighted, 'val': valid_loader, 'test': test_loader}
            else:
                test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
                dataloaders = {'train': train_loader_weighted, 'val': val_loader_weighted, 'test': test_loader}
                print("Weighted Dataset Loaders Created")

            if display_weighted_classes:
                num = 0
                class_count_weighted = [0] * 31
                for x, y in train_loader_weighted:
                    num = num + 1
                    print("batch num", num)
                    for cls in range(31):
                        index = (y == cls).sum().item()
                        class_count_weighted[cls] += index
                print("train loader weighted", class_count_weighted)

        return dataloaders

    @staticmethod
    def define_transforms_train(img_size, mean, std):
        """
        define the necessary image transformations for training data
        :param img_size: resize to this size
        :param mean: dataset mean
        :param std: dataset std deviatioin
        :return: image transformations
        """

        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            # transforms.RandomResizedCrop(size=315, scale=(0.95, 1.0)),
            # transforms.RandomRotation(degrees=5),
            transforms.RandomHorizontalFlip(p=.3),
            # transforms.CenterCrop(size = 299),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=.4, hue=.1, saturation=.2, contrast=.2),
            ], p=.3),
            transforms.RandomApply([
                transforms.GaussianBlur(5, 1)
                # transforms.CenterCrop(size=400)
            ], p=.3),
            transforms.RandomApply([
                transforms.RandomRotation(degrees=10),
            ], p=.3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return train_transform

    @staticmethod
    def define_transforms_test(img_size, mean, std):
        """
        define the necessary image transformations for test data
        :param img_size: resize to this size
        :param mean: dataset mean
        :param std: dataset std deviatioin
        :return: image transformations
        """
        test_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return test_transforms

    @staticmethod
    def plot_data(data, encoder, num_fig=12, inv_normalize=None):
        """
        function to plot random samples from the dataset
        :param data: dataset to plot
        :param encoder: class id to name mapping dictionary
        :param num_fig: num of figures to plot
        :return: plot the images
        """
        num_row = int(num_fig / 4)
        fig, axes = plt.subplots(figsize=(14, 10), nrows=num_row, ncols=4)
        for ax in axes.flatten():
            k = random.randint(0, len(data))
            (image, label) = data[k]
            label = int(label)
            if inv_normalize is not None:
                image = inv_normalize(image)
            class_name = encoder[label]
            image = image.numpy().transpose(1, 2, 0)
            im = ax.imshow(image)
            ax.set_title(class_name)
            ax.axis('off')
        plt.show()

    @staticmethod
    def inv_normalize(mean, std):
        """
        do the inverse normalization for visualization purposes
        :param mean: mean of the dataset
        :param std: std deviation of the dataset
        :return: inverse normalization
        """
        return transforms.Normalize(
            mean=-1 * np.divide(mean, std),
            std=1 / std
        )

    @staticmethod
    def decode_encode(classes):
        """
        get the mapping between class names and corresponding integer numbers
        :param classes: classes in the dataset
        :return: both mappings
        """
        decoder = {}
        encoder = {}
        for i in range(len(classes)):
            decoder[classes[i]] = i
        for i in range(len(classes)):
            encoder[i] = classes[i]
        return decoder, encoder

    @staticmethod
    def get_encoder(classes):
        """
        get the mapping between class names and corresponding integer numbers
        :param classes: classes in the dataset
        :return: mappings
        """
        encoder = {}
        for i in range(len(classes)):
            encoder[i] = classes[i]
        return encoder

    @staticmethod
    def keep_aspectratio(net_w, net_h, image_orig, channel=3):
        """
        resize the image keeping the aspect ratio
        :param net_w: width
        :param net_h: height
        :param image_orig: original image
        :param channel: channel color or grey
        :return: resized image
        """
        im_h, im_w, _ = image_orig.shape
        if channel > 1:
            new_image = np.zeros((net_h, net_w, channel), dtype=np.uint8)
        else:
            new_image = np.zeros((net_h, net_w), dtype=np.uint8)
        if net_w / im_w <= net_h / im_h:
            image_orig = cv2.resize(image_orig, (net_w, int(im_h * (net_w / im_w))))
            new_image[int((net_h - int(im_h * (net_w / im_w))) / 2):int(
                (net_h - int(im_h * (net_w / im_w))) / 2) + int(im_h * (net_w / im_w)), :] = image_orig
        else:
            image_orig = cv2.resize(image_orig, (int(im_w * (net_h / im_h)), net_h))
            new_image[:, int((net_w - int(im_w * (net_h / im_h))) / 2):int(
                (net_w - int(im_w * (net_h / im_h))) / 2) + int(im_w * (net_h / im_h))] = image_orig

        return new_image
