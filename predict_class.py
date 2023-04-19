__author__ = "gpillai"

import time
import os
from PIL import Image
import numpy as np
import cv2
import torch
import yaml
from torch import nn
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from dataset import ClassifierData
from model.classifier import Classifier


class ClassPredictor:
    def __init__(self, weights_dir, checkpoint, out_dir='./', classes=None, img_size=400, mean=None, std=None, model=None,
                 model_type='state_dict'):
        if mean is None:
            mean = [.5, .5, .5]
        if std is None:
            std = [.3, .3, .3]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dir = weights_dir
        self.checkpoint = checkpoint
        self.classes = classes
        self.out_dir = out_dir

        if model_type == 'full_model':
            self.model = torch.load(os.path.join(self.weights_dir, self.checkpoint))
        else:
            if model is None:
                model = {"name": 'resnet', "version": '50'}
            self.model = Classifier(len(self.classes), model)
            checkpoint_ = torch.load(os.path.join(self.weights_dir, self.checkpoint))
            self.model.load_state_dict(checkpoint_['model_state_dict'])

        self.model.to(self.device)
        self.model.eval()
        self.encoder = ClassifierData.get_encoder(self.classes)
        # compute the mean and std for your dataset if not known as shown in the train_classifier.py
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.img_size = img_size
        # define the transformations required for the image
        self.transforms = ClassifierData.define_transforms_test(self.img_size, self.mean, self.std)
        # do the inverse normalization for visualization purposes
        # self.inv_normalize = ClassifierData.inv_normalize(self.mean, self.std)

    def prediction_bar(self, output, print_results, show_conf_bar):
        """
		display the bar with confidence score for top 5 classes
		:param output: predicted output
		:param print_results: print the predictions on to console
		:param show_conf_bar: Display the plot for confidence scores
		:return label: label of the class
		:return confidence: confidence of the prediction
		"""
        output = output.cpu().detach().numpy()
        a = output.argsort()
        a = a[0]

        size = len(a)
        if size > 5:
            a = np.flip(a[-5:])
        else:
            a = np.flip(a[-1 * size:])
        if print_results:
            for i in a:
                print('Class: {} , confidence: {}'.format(self.encoder[int(i)], float(output[:, i] * 100)))
                print('Class: {}'.format(self.encoder[int(i)]))
        if show_conf_bar:
            prediction = list()
            classes = list()
            for i in a:
                prediction.append(float(output[:, i] * 100))
                classes.append(str(i))
            plt.bar(classes, prediction)
            plt.title("Confidence score bar graph")
            plt.xlabel("Confidence score")
            plt.ylabel("Class number")
            plt.show()
        label = self.encoder[a[0]]
        confidence = round(float(output[:, a[0]] * 100), 2)
        return label, confidence

    @staticmethod
    def show_image(image, label, confidence):
        """
		displays the image
		:param image: image
		:param label: label of the class
        :param confidence: confidence of the prediction
        """
        cv2.putText(image, "Label: " + label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 127, 0), 2)
        cv2.putText(image, "Conf: " + str(confidence) + " %", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 127, 0), 2)
        cv2.imshow("prediction", image)
        cv2.waitKey(0)

    def save_image(self, image, label, confidence, img_name):
        """
        saves the image with its prediction
        :param image: image to save
        :param label: label of the class
        :param confidence: confidence of the prediction
        :param img_name: image name to save the result
        """
        cv2.putText(image, "Label: " + label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 127, 0), 2)
        cv2.putText(image, "Conf: " + str(confidence) + " %", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 127, 0), 2)
        cv2.imwrite(os.path.join(self.out_dir, img_name), image)

    def predict(self, image, img_name, transforms=True, keep_aspect=True, save_image=True, show_image=False, print_results=True,
                show_conf_bar=False):
        """
		predict the label for the image using the trained model. This method will resize the image into the required size.
		It is recommended that the bbox of the object predicted from an object detector is given an offset to all directions (for eg. 50px).
		:param image: image to predict on
		:param img_name: image name
		:param transforms: transformations for image
		:param save_image: Indicate whether to save the image to the out directory
		:param show_image: Indicate whether to display the image after prediction
		:param print_results: Indicate whether to print the results on console
		:param show_conf_bar: show the plot for confidence predictions
		:return: predictions
		"""
        vis_image = image
        if keep_aspect:
            image = ClassifierData.keep_aspectratio(self.img_size, self.img_size, image)
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        if transforms:
            image = self.transforms(image)

        data = image.expand(1, -1, -1, -1)
        data = data.type(torch.FloatTensor).to(self.device)
        soft = nn.Softmax(dim=1)
        start = time.time()
        output = self.model(data)
        end = time.time()
        fps = round(1 / (end - start))
        print("FPS: ", fps)
        output = soft(output)

        label, confidence = self.prediction_bar(output, print_results, show_conf_bar)

        if save_image:
            self.save_image(vis_image, label, confidence, img_name)
        if show_image:
            self.show_image(vis_image, label, confidence)

        return label, confidence


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config', default='./config/config_color_predict.yaml', help='path to prediction config file')
    args = parser.parse_args()

    config_file = open(args.config, 'r')
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    # initialize the necessary path variables, model name etc
    classes = config['classes']
    img_size = config['img_size']
    keep_aspect = config['keep_aspect']
    weights_dir = config['weights_dir']
    checkpoint = config['checkpoint']
    model_type = config['model_type']
    model = config['model']
    img_path = config['img_path']
    out_dir = config['out_dir']
    do_transforms = config['do_transforms']
    mean = config['mean']
    std = config['std']

    # initialize the class Predictor object
    predict_obj = ClassPredictor(weights_dir, checkpoint, out_dir, classes, img_size, mean, std, model, model_type)

    for image_name in os.listdir(img_path):
        frame = cv2.imread(os.path.join(img_path, image_name))
        print("Image : ", image_name)
        # predict on the input image and get the result
        label, confidence = predict_obj.predict(frame, image_name, do_transforms, keep_aspect)
        print("\n")
