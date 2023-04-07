#  Copyright (c) Ikara Vision Systems GmbH, Kaiserslautern - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential

"""
Functions to evaluate different metrics and to help logging in tensorboard
"""

import tqdm
from sklearn import metrics
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


def compute_mean_std(train_loader):
    """
    compute the mean and std deviation for our dataset
    :param train_loader: dataloader for training set
    :return: mean and std deviation
    """
    mean = 0
    std = 0
    num_samples = len(train_loader.dataset)
    for data, _ in tqdm.tqdm(train_loader):
        batch_sample = data.size(0)
        data = data.view(batch_sample, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
    mean /= num_samples
    std /= num_samples
    return mean.numpy(), std.numpy()


def get_norm_params(batch_size, lp_train, img_size):
    """
    function to compute the mean and std parameters
    :param batch_size: batch size for training
    :return: mean and std deviation
    """
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()])

    train_data = torchvision.datasets.ImageFolder(root=lp_train,
                                                  transform=train_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    mean, std = compute_mean_std(train_loader)
    return mean, std


def pr_curve_tensorboard(classes, class_index, test_probs, test_preds, writer, global_step=1):
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)


def perf_matrix(true, pred):
    """
    compute the performace matrix
    :param true: true label
    :param pred: predicted label
    """
    precision = metrics.precision_score(true, pred, average='macro')
    recall = metrics.recall_score(true, pred, average='macro')
    accuracy = metrics.accuracy_score(true, pred)
    f1_score = metrics.f1_score(true, pred, average='macro')
    print('Precision: {} Recall: {}, Accuracy: {}: ,f1_score: {}'.format(precision * 100, recall * 100, accuracy * 100,
                                                                         f1_score * 100))


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    print("weight per class: ", weight_per_class)
    return weight