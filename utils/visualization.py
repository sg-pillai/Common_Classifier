"""
Functions to help different visualizations
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from sklearn import metrics
import torch.nn.functional as F


def plot_loss(loss):
	"""
	plot the loss curve
	:param loss: loss from training
	"""
	plt.figure(figsize=(10, 5))
	plt.plot(loss)
	plt.title("Training loss")
	plt.xlabel("epochs")
	plt.ylabel("Loss")
	plt.show()


def plot_accuracy(accuracy):
	"""
	plot the accuracy curve
	:param accuracy: accuracy from training
	"""
	plt.figure(figsize=(10, 5))
	plt.plot(accuracy)
	plt.title("Training accuracy plot")
	plt.xlabel("epochs")
	plt.ylabel("accuracy")
	plt.show()


def plot_wrong(num_fig, true, image, pred, encoder, inv_normalize):
	"""
	Plot the wrong predictions from the network
	:param num_fig: number of figures to plot
	:param true: true label
	:param image: image
	:param pred: predicted label
	:param encoder: mapping of integer number to classes
	:param inv_normalize: inverse normalization for the image
	"""
	n_row = int(num_fig / 3)
	fig, axes = plt.subplots(figsize=(14, 10), nrows=n_row, ncols=3)
	for ax in axes.flatten():
		a = random.randint(0, len(true) - 1)
		image, correct, wrong = image[a], true[a], pred[a]
		image = torch.from_numpy(image)
		print("image shape ", image.shape)
		correct = int(correct)
		c = encoder[correct]
		print("class", c)
		wrong = int(wrong)
		w = encoder[wrong]
		f = 'A:' + c + ',' + 'P:' + w
		# if inv_normalize != None:
		#     image = inv_normalize(image)
		image = image.numpy().transpose(1, 2, 0)
		im = ax.imshow(image)
		ax.set_title(f)
		ax.axis('off')
	plt.show()


def plot_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
	"""
	plot the confusion matrix
	:param y_true: true labels
	:param y_pred: predicted labels
	:param classes: classes in the dataset
	:param normalize: indicate whether to normalize or not
	:param title: title for the plot
	:param cmap: cmap for the plot
	:return: plot
	"""
	if not title:
		if normalize:
			title = 'Normalized confusion matrix'
		else:
			title = 'Confusion matrix, without normalization'
	# Compute confusion matrix
	cm = metrics.confusion_matrix(y_true, y_pred)
	# Only use the labels that appear in the data
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)

	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   # ... and label them with the respective list entries
		   xticklabels=classes, yticklabels=classes,
		   title=title,
		   ylabel='True label',
		   xlabel='Predicted label')

	# # ax.set(xticks=12, yticks=12, xticklabels=classes, yticklabels=classes, title=title, ylabel='True label', xlabel='Predicted label')
	# ax.set(xticklabels=classes, yticklabels=classes, title=title, ylabel='True label',
	# 	   xlabel='Predicted label')

	print("after setting plot")
	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	plt.savefig('matrix.png')
	plt.show()

	return ax


def matplotlib_imshow(img, one_channel=False):
	"""
	Helper function to plot the images into the figure
	:param img: image
	:param one_channel: rgb or gray
	"""
	if one_channel:
		img = img.mean(dim=0)
	# img = img / 2 + 0.5  # unnormalize
	img1 = (img * 0.26404583) + 0.14058352  # use the respective mean and std here
	img2 = img1 * 255
	img2 = img2.astype(np.uint8)
	# npimg = img.cpu().numpy()
	if one_channel:
		plt.imshow(img, cmap="Greys")
	else:
		plt.imshow((np.transpose(img2, (1, 2, 0))))
	# plt.close()


def plot_predictions(output, images, labels, classes):
	"""
	helper function to plot predictions vs target into tensorboard
	:param output: prediction
	:param images: target image
	:param labels: target label
	:param classes: classes name
	:return: figure
	"""
	_, preds_tensor = torch.max(output, 1)
	preds = np.squeeze(preds_tensor.cpu().numpy())
	images = images.cpu().numpy()
	confidence = [F.softmax(pd, dim=0)[i].item() for i, pd in zip(preds, output)]
	num_imgs = len(confidence)
	fig = plt.figure(figsize=(10, 2 * num_imgs))
	print('plot_predictions: Number of imgs: ', num_imgs, num_imgs/2, int(num_imgs/2))
	for id in range(num_imgs):
		ax = fig.add_subplot(int(num_imgs/2), 2, id + 1, xticks=[], yticks=[])
		matplotlib_imshow(images[id], one_channel=False)
		ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
			classes[preds[id]],
			confidence[id] * 100.0,
			classes[labels[id]]),
			color=("green" if preds[id] == labels[id].item() else "red"))

	plt.close(fig)

	return fig
