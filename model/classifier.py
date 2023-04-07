#  Copyright (c) Ikara Vision Systems GmbH, Kaiserslautern - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential

__author__ = "gpillai"

from torch import nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


class Classifier(nn.Module):
	"""
	Classifier class. choose the model accordingly (resnet or efficientnet)
	"""

	def __init__(self, num_classes, model):
		"""
		initialize our classifier
		:param num_classes: number of classes in our dataset
		:param model: chosen model - resnet or efficientnet
		"""
		super(Classifier, self).__init__()
		if model["name"] == 'resnet':
			if model["version"] == '50':
				print("model is resnet50")
				self.model = models.resnet50(pretrained=True)
			elif model["version"] == '101':
				print("model is resnet101")
				self.model = models.resnet101(pretrained=True)
			elif model["version"] == '152':
				print("model is resnet152")
				self.model = models.resnet152(pretrained=True)

		elif model["name"] == 'mobilenet':
			if model["version"] == '2':
				print("model is mobilenet version 2")
				self.model = models.mobilenet_v2(pretrained=True)
			if model["version"] == '3':
				print("model is mobilenet version 3")
				self.model = models.mobilenet_v3_large(pretrained=True)

		elif model["name"] == 'resnext':
			if model["version"] == '50':
				print("model is ResNeXt-50 32x4d")
				self.model = models.resnext101_32x8d(pretrained=True)
			else:
				print("model is ResNeXt-101 32x8d")
				self.model = models.resnext101_32x8d(pretrained=True)

		elif model["name"] == 'wide_resnet':
			if model["version"] == '50':
				print("model is Wide ResNet-50-2")
				self.model = models.wide_resnet50_2(pretrained=True)
			else:
				print("model is Wide ResNet-101-2")
				self.model = models.wide_resnet101_2(pretrained=True)

		else:
			print("model is efficientnet")
			model_name = model["name"] + model["version"]
			self.model = EfficientNet.from_pretrained(model_name)

		self.l1 = nn.Linear(1000, 512)
		self.l2 = nn.Linear(512, 256)
		self.l3 = nn.Linear(256, 128)
		self.l4 = nn.Linear(128, num_classes)
		self.dropout = nn.Dropout(0.5)
		self.relu = nn.ReLU()

	def forward(self, input):
		"""
		define the forward function for our network. Implement the final layers for the model selected
		:param input: input data
		:return: final network output
		"""
		x = self.model(input)
		x = x.view(x.size(0), -1)
		x = self.dropout(self.relu(self.l1(x)))
		x = self.dropout(self.relu(self.l2(x)))
		x = self.dropout(self.relu(self.l3(x)))
		x = self.l4(x)
		return x
