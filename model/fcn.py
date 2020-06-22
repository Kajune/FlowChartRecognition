import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FCN(nn.Module):
	def __init__(self, n_classes):
		super().__init__()
		self.net = models.segmentation.fcn_resnet50(pretrained=False, num_classes=n_classes, progress=False)

	def forward(self, x):
		return self.net(x)['out']

	def predict(self, x):
		return self.forward(x)

class DeepLabV3(nn.Module):
	def __init__(self, n_classes):
		super().__init__()
		self.net = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=n_classes, progress=False)

	def forward(self, x):
		return self.net(x)['out']

	def predict(self, x):
		return self.forward(x)
