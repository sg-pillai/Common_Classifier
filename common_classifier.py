__author__ = "gpillai"
"""
Common functions for the purposes of training testing and evaluation
"""
import torch
import time
import os
import numpy as np
from torch.autograd import Variable
import torch.optim as torch_optimiser
from torch import nn
import torch.nn.functional as F

from utils.early_stopping import EarlyStopping
from utils.visualization import plot_predictions
from utils.evaluate import pr_curve_tensorboard


def train(model, dataloaders, criterion, optimiser, lr_scheduler, weight_dir, classes, writer, start_epoch=0,
		  num_epocs=10, batch_size=8, early_steps=None):
	"""
	define the training steps
	:param model: model for training
	:param dataloaders: datalosder for the dataset
	:param criterion: loss
	:param optimiser: optimiser for the model
	:param lr_scheduler: learning rate scheduler
	:param weight_dir: directory to store the weights
	:param classes: Class names of the images
	:param writer: tensorboard writer object
	:param start_epoch: starting epoch of this training experiment
	:param num_epocs: number of epochs to train
	:param batch_size: batch size
	:param early_steps: number of epochs to consider for earling stopping
	:return: loss and accuracy
	"""
	tic = time.time()
	phases = dataloaders.keys()
	losses = list()
	accuracy = list()
	val_losses = list()
	val_accuracy = list()

	if early_steps is not None:
		earlystop = EarlyStopping(wait_epoch_num=early_steps, print_loss=True, weight_dir=weight_dir)

	for epoch in range(start_epoch, num_epocs):
		print("Epoch Number:", epoch)

		for phase in phases:
			if phase == 'train':
				model.train()
			else:
				model.eval()
			running_loss = 0
			correct = 0
			j = 0
			for batch_id, (data, target) in enumerate(dataloaders[phase]):
				len_ds = len(dataloaders[phase])
				data, target = Variable(data), Variable(target)
				data = data.type(torch.cuda.FloatTensor)
				target = target.type(torch.cuda.LongTensor)
				optimiser.zero_grad()
				output = model(data)

				loss = criterion(output, target)
				_, preds = torch.max(output, 1)
				correct = correct + torch.sum(preds == target.data)
				running_loss = running_loss + (loss.item() * data.size(0))
				j = j + 1

				if phase == 'train':
					loss.backward()
					optimiser.step()

				if phase == 'val':
					if batch_id < len(dataloaders[phase]) - 1 and len(target) == batch_size and batch_id == len_ds - 2:
						fig = plot_predictions(output, data, target, classes)
						writer.add_figure('Predictions vs GroundTruth', fig, global_step=epoch)

				if batch_id % 10 == 0:
					print("{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAcc: {:.6f}".format(phase, epoch, batch_id * len(data), len(
							dataloaders[phase].dataset), 100 * batch_id / len(dataloaders[phase]), running_loss / (
																									  j * batch_size),
																							  correct.double() / (
																									  j * batch_size)))
				del data
				del target
				del output
				del preds

				# break

			epoch_accuracy = correct.double() / (len(dataloaders[phase]) * batch_size)
			epoch_loss = running_loss / (len(dataloaders[phase]) * batch_size)
			writer.add_scalar(phase + 'Loss', epoch_loss, epoch)
			writer.add_scalar(phase + 'Accuracy', epoch_accuracy, epoch)
			writer.add_scalar('lr_value', lr_scheduler.get_last_lr()[0], epoch)
			print(phase + " Loss: ", epoch_loss)

			if phase == 'val':
				val_losses.append(epoch_loss)
				val_accuracy.append(epoch_accuracy)
				if early_steps is not None:
					earlystop(epoch_loss, model, str(epoch))
				print("Early Stop TRUE or FALSE", earlystop.earlystop)

			if phase == 'train':
				losses.append(epoch_loss)
				accuracy.append(epoch_accuracy)
				print("{} Accuracy: ".format(phase), epoch_accuracy.item())
				print("Training time for this epoch{}".format(time.time() - tic))

				torch.save({
					'epoch': epoch,
					'model_state_dict': model.state_dict(),
					'optimizer': optimiser.state_dict(),
				}, os.path.join(weight_dir, "checkpoint_" + str(epoch) + ".pt"))

		# break
		lr_scheduler.step()
		del loss
		del epoch_loss
		del running_loss
		del epoch_accuracy

		if earlystop.earlystop:
			print("Early Stopping !!")
			# model.load_state_dict(torch.load('./checkpoint.pt'))
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer': optimiser.state_dict(),
			}, os.path.join(weight_dir, "checkpoint_Early_Stopped_" + str(epoch) + ".pt"))
			break

	return losses, accuracy, val_losses, val_accuracy


def test(model, dataloader, criterion, batch_size, writer, classes):
	"""
	test function to test on test data
	:param dataloader: dataloader
	:return: returns correctly predicted values and wrongly predicted ones and the corresponding image for verification
	"""
	correct = 0
	running_loss = 0
	pred = []
	corr = []
	wrong_pred = []
	wrong_corr = []
	image = []
	soft = nn.Softmax(dim=1)
	class_probs = []
	class_preds = []
	model.eval()
	with torch.no_grad():
		for batch_id, (data, target) in enumerate(dataloader):
			len_ds = len(dataloader)
			data, target = Variable(data), Variable(target)
			data = data.type(torch.cuda.FloatTensor)
			target_ = target.type(torch.cuda.LongTensor)
			output = model(data)
			loss = criterion(output, target_)
			output_prob = soft(output)
			# loss.detach()
			_, prediction = torch.max(output_prob, 1)
			correct = correct + torch.sum(prediction == target_.data).detach()
			running_loss += loss.item() + data.size(0)
			prediction = prediction.detach().cpu()
			prediction_ = prediction.numpy()
			target_ = target_.detach().cpu().numpy()
			target = target_
			preds = np.reshape(prediction_, (len(prediction_), 1))
			target = np.reshape(target, (len(preds), 1))
			# data = data.cpu().numpy()

			if batch_id == len_ds - 2:
				writer.add_figure('Predictions vs GroundTruth Testing', plot_predictions(output, data, target_, classes),
								  global_step=1)

			class_probs_batch = [F.softmax(el, dim=0) for el in output]
			# output = output.detach()
			# _, class_preds_batch = torch.max(output, 1)
			# output_prob = output_prob.detach().cpu().numpy()
			class_probs.append(class_probs_batch)
			class_preds.append(prediction)

			for i in range(len(preds)):
				pred.append((preds[i]))
				corr.append((target[i]))
				if preds[i] != target[i]:
					wrong_pred.append(preds[i])
					wrong_corr.append(target[i])
					image.append(data[i])

			del data
			del target, target_
			del preds
			del prediction_, prediction
			del class_probs_batch
			del output
			del loss
			# break

	test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
	test_preds = torch.cat(class_preds)

	for i in range(len(classes)):
		pr_curve_tensorboard(classes, i, test_probs, test_preds, writer)

	epoch_accuracy = correct.double() / (len(dataloader) * batch_size)
	epoch_loss = running_loss / (len(dataloader) * batch_size)
	writer.add_scalar('Test Loss', epoch_loss, 1)
	writer.add_scalar('Test Accuracy', epoch_accuracy, 1)

	print(" Epoch accuracy and loss: ", epoch_accuracy, epoch_loss)

	del epoch_accuracy
	del epoch_loss

	return corr, pred, image, wrong_corr, wrong_pred
