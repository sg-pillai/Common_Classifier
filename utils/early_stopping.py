import os
import numpy as np
import torch


class EarlyStopping:
	"""
    Class to define the early stopping
    it stops the training when the validation loss does not improve after a given time
    """

	def __init__(self, wait_epoch_num=7, print_loss=False, weight_dir="./weights"):
		"""
        Initialise the necessary parameters
        :param wait_epoch_num: Number of epochs to wait after validation lost has last improved
        :param print_loss: Make it True to print the val loss each time it is improved
        """
		self.wait_epoch_num = wait_epoch_num
		self.print_loss = print_loss
		self.counter = 0
		self.best_score = None
		self.earlystop = False
		self.min_val_loss = np.inf
		self.dir = weight_dir

	def __call__(self, val_loss, model, epoch):
		"""
        if val loss is not improving for the required number of epochs stop the training and save the model.
        Save the model for best score
        :param val_loss: validation loss
        :param model: model
        """
		score = -val_loss
		if self.best_score is None:
			self.best_score = score
			# self.checkpoint_save(val_loss, model, epoch)
		elif score < self.best_score:
			self.counter += 1
			print("Early Stopping counter value {} / {}".format(self.counter, self.wait_epoch_num))
			if self.counter >= self.wait_epoch_num:
				self.earlystop = True
		else:
			self.best_score = score
			# uncomment the below code if you want to save the best model
			# self.checkpoint_save(val_loss, model, epoch)
			self.counter = 0

	def checkpoint_save(self, val_loss, model, epoch):
		"""
        Save the model when the val score is best
        :param val_loss: val loss
        :param model: model
        """
		if self.print_loss:
			print("Validation loss decreased from {:.6f} to {:.6f}. Hence saving the model...".format(self.min_val_loss,
																									  val_loss))
		torch.save(model.state_dict(), os.path.join(self.dir, 'checkpoint_' + str(epoch) + '.pt'))
		self.min_val_loss = val_loss
