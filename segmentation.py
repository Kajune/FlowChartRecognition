import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torchsummary import summary

import model.focalLoss as fl
from model.unet import UNet, UNetPP
from model.fcn import *

from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='UNet', choices=['FCN', 'DeepLab', 'UNet', 'UNetPP'])
args = parser.parse_args()

if __name__ == '__main__':
	size=(384,384)

	loader_train = torch.utils.data.DataLoader(FlowchartDataset(set='train', size=size, osm_alpha='random', aug=False), 
		batch_size=2, shuffle=True)
	loader_test = torch.utils.data.DataLoader(FlowchartDataset(set='test', size=size, osm_alpha='random'), batch_size=2, shuffle=False)

	device = ('cuda' if torch.cuda.is_available() else 'cpu')

	if args.model == 'FCN':
		model = FCN(n_classes=len(class_names))
	elif args.model == 'DeepLab':
		model = DeepLabV3(n_classes=len(class_names))
	elif args.model == 'UNet':
		model = UNet(n_channels=3, n_classes=len(class_names))
	elif args.model == 'UNetPP':
		model = UNetPP(n_channels=3, n_classes=len(class_names))

	model = model.to(device)
#	summary(model, (3, size[1], size[0]))

	optimizer = optim.Adam(model.parameters())
#	criteria = nn.BCEWithLogitsLoss()
	criteria = fl.FocalLossBCE(gamma=2.0)

	os.makedirs('result/' + args.model, exist_ok=True)

	for epoch in range(20):
		for i, (img, gt) in enumerate(loader_train):
			img = img.to(device)
			gt = gt.to(device)

			optimizer.zero_grad()
			loss = criteria(model(img), gt)
			loss.backward()
			optimizer.step()

			print('[%d, %d/%d] loss: %f' % (epoch, i, len(loader_train), loss.item()))

		model.eval()
		with torch.no_grad():
			iou = []
			for i, (img, gt) in enumerate(loader_test):
				input = img.numpy().transpose((0,2,3,1))
				answer = gt.numpy().transpose((0,2,3,1))

				img = img.to(device)
				result = model.predict(img).cpu().numpy().transpose((0,2,3,1))

				for j in range(result.shape[0]):
					iou.append(computeIoU(result[j], answer[j]))
					cv2.imwrite('result/%s/epoch%03d_%d_%d.png' % (args.model, epoch, i, j), 
						np.hstack((input[j] * 255, visualizeMap(result[j]), visualizeMap(answer[j]))))

			write = open('result/%s/epoch%d.txt' % (args.model, epoch), 'w')
			iou = np.array(iou)
			print('IoU')
			write.write('IoU\n')
			mean = np.mean(iou, axis=0)
			for i, c in enumerate(class_names):
				print('\t' + c + ': %f' % mean[i])
				write.write('\t' + c + ': %f\n' % mean[i])
			print('mIoU: %f' % (np.mean(iou)))
			write.write('mIoU: %f\n' % (np.mean(iou)))
			write.close()

		model.train()
