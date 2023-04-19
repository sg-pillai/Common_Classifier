__author__ = "gpillai"

import sys
import os
from datetime import datetime
import numpy as np
import yaml
from argparse import ArgumentParser
import traceback

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.optim as torch_optimiser
from torch_lr_finder import LRFinder

from dataset import ClassifierData
from model.classifier import Classifier
from common_classifier import train, test
from utils.visualization import plot_loss, plot_accuracy, plot_matrix
from utils.evaluate import perf_matrix, get_norm_params, make_weights_for_balanced_classes


def train_model(model, dataloaders, criterion, optimiser, lr_scheduler, weight_dir, writer, epoch, num_epochs=10,
                batch_size=8,
                early_steps=None, classes=None, plot_results=False):
    """
	train the model and plot the necessary curves

	:param model: model
	:param dataloaders: dataloader for the dataset
	:param criterion: loss function
	:param optimiser: optimiser for the model
	:param lr_scheduler: learning rate scheduler
	:param weight_dir: path to save the weight files
	:param writer: tensorboard writer
	:param epoch: current epoch
	:param num_epochs: number of epochs
	:param batch_size: batch size
	:param early_steps: number of epochs to check for early stopping
	:param classes: number of classes
	:param plot_results: indicate whether to plot the accuracy loss etc.
	"""
    try:
        train_loader = {}
        do_test = False
        phases = dataloaders.keys()
        for phase in phases:
            if phase == 'test':
                do_test = True
            else:
                train_loader.update([(phase, dataloaders[phase])])
        losses, accuracy, val_losses, val_accuracy = train(model, train_loader, criterion, optimiser, lr_scheduler,
                                                           weight_dir, classes, writer, epoch, num_epochs, batch_size,
                                                           early_steps)

        if plot_results:
            plot_loss(losses)
            plot_accuracy(accuracy)
            plot_loss(val_losses)
            plot_accuracy(val_accuracy)

        if do_test:
            correct, pred, image, wrong_corr, wrong_pred = test(model, dataloaders['test'], criterion, batch_size,
                                                                writer, classes)
            # plot_wrong(12, wrong_corr, image, wrong_pred, encoder, inv_normalize)
            perf_matrix(correct, pred)
            if classes is not None:
                plot_matrix(correct, pred, classes)

        return True

    except Exception as e:
        print("Error occured during runtime: \n", e)
        traceback.print_exc()
        return False


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config', default='./config/config_car.yaml', help='path to config file')
    args = parser.parse_args()

    config_file = open(args.config, 'r')
    config = yaml.load(config_file)
    batch_size = config["batch_size"]
    img_size = config["img_size"]
    model = config["model"]
    num_epochs = config["num_epochs"]
    lr = config["lr"]['value']
    find_lr = config["lr"]["find_lr"]
    step_size = config['lr']['step_size']
    gamma = config['lr']['gamma']

    early_steps = config["early_steps"]
    weight_dir = config["weight_dir"]

    mean = config["mean"]
    std = config["std"]

    visualize_random_images = config["config_options"]["visualize_random_images"]

    # give the path to train and test data to the ImageFolder
    lp_train = config["train_data"]
    lp_val = config["val_data"]
    lp_test = config["test_data"]
    balance_dataset = config['balance_dataset']
    epoch = 0

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    if mean is None or std is None:
        mean, std = get_norm_params(batch_size, lp_train, img_size)

    # mean, std = np.array([0.53304017, 0.53191423, 0.5446271]), np.array([0.32864863, 0.3216072, 0.31737655])
    # mean = np.array([0.14058352, 0.14319709, 0.14595295])
    # std = np.array([0.26404583, 0.26498058, 0.26745948])
    # [0.42095104 0.408939   0.40319932] [0.30645183 0.3025921  0.3009797 ]

    print("Mean and Std of dataset: ", mean, std)

    # define the necessary image transformations
    train_transforms = ClassifierData.define_transforms_train(img_size, mean, std)
    test_transforms = ClassifierData.define_transforms_test(img_size, mean, std)

    train_data = torchvision.datasets.ImageFolder(root=lp_train, transform=train_transforms)
    val_data = torchvision.datasets.ImageFolder(root=lp_val, transform=test_transforms)
    test_data = torchvision.datasets.ImageFolder(root=lp_test, transform=test_transforms)

    # create the dataloader using the above dataset paths
    dataloaders = ClassifierData.data_loader(train_data, val_data, test_data, test_size=0.01, batch_size=batch_size,
                                             balance_dataset=balance_dataset)

    # convert classes into integer to  make it easy to find labels
    classes = train_data.classes
    print("classes for training:", classes)
    decoder, encoder = ClassifierData.decode_encode(classes)

    if visualize_random_images:
        # Use inverse normalize if we have already normalized the original image (for visualization purpose)
        inv_normalize = ClassifierData.inv_normalize(mean, std)
        ClassifierData.plot_data(train_data, encoder, 12, inv_normalize)

    # get the available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize the network classifier with the required model and number of classes in out dataset
    classifier = Classifier(len(classes), model)
    classifier.to(device)

    # define the criterion for the network
    criterion = nn.CrossEntropyLoss()

    optimiser = torch_optimiser.Adam(classifier.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=step_size, gamma=gamma)

    # find the best learning rate using the lr finder
    # for more info refer https://github.com/davidtvs/pytorch-lr-finder
    # execute the lr finder and plot the results
    if find_lr:
        lr_f = LRFinder(classifier, optimiser, criterion, device=device)
        lr_f.range_test(dataloaders["train"], end_lr=1, num_iter=500)
        lr_f.reset()
        lr_f.plot()

    # check if to resume training
    if config["resume"]["status"]:
        checkpoint = torch.load(config["resume"]["path"])
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict((checkpoint['optimizer']))
        epoch = checkpoint["epoch"]
        weight_dir = config["resume"]["log"]
        checkpoint_dir = os.path.join(weight_dir, 'checkpoint')
        tensorboard_dir = os.path.join(weight_dir, 'tensorboard')
    else:
        # save the configuration for the current training experiments
        now = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        weight_dir = os.path.join(weight_dir, config["model"]["name"] + config["model"]["version"] + "-" + now)
        checkpoint_dir = os.path.join(weight_dir, 'checkpoint')
        tensorboard_dir = os.path.join(weight_dir, 'tensorboard')

    if not os.path.isdir(weight_dir):
        os.makedirs(weight_dir)
        os.makedirs(checkpoint_dir)
        os.makedirs(tensorboard_dir)

    writer = SummaryWriter(tensorboard_dir)

    with open(os.path.join(weight_dir, "config_.yaml"), "w") as f:
        yaml.dump(config, f)

    # # start training
    status = train_model(classifier, dataloaders, criterion, optimiser, lr_scheduler, checkpoint_dir, writer, epoch,
                         num_epochs, batch_size,
                         early_steps, classes)

    print("Training Finished !!")
    writer.close()
    sys.exit()
