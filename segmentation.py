import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary

import flowchart
import focalLoss as fl
from model.unet import UNet
from model.fcn import *

class_names = ['process', 'decision', 'terminator', 'data', 'connection', 'arrow', 'text']
fill_flag = [True, True, True, True, True, False, False]
color = [(0,0,255), (0,255,0), (255,255,0), (0,255,255), (255,0,255), (255,0,0), (255,255,255)]

def imshow(img, title='', scale=0.25):
	cv2.imshow(title, cv2.resize(img, None, fx=scale, fy=scale))

def visualizeMap(map):
	img = np.zeros((map.shape[0], map.shape[1], 3), dtype=np.uint8)
	for i in range(map.shape[2]):
		img[map[:,:,i] > 0] = color[i]
	return img

def makeImages(linewidth=5):
	dataset = flowchart.load()

	imgList = []

	for i, data in enumerate(dataset):
		print('\r%d/%d' % (i, len(dataset)), end='')

		img = np.ones((int(data['size'][1]), int(data['size'][0])), dtype=np.float32)
		gt = np.zeros((int(data['size'][1]), int(data['size'][0]), len(class_names)), dtype=np.uint8)

		for c, coords in enumerate(data['coords']):
			cls_id = class_names.index(data['annot'][c][0])
			fill = fill_flag[cls_id]

			for coord in coords:
				pts = coord.astype(np.int32)
				pts = pts.reshape((-1,1,2))
				cv2.polylines(img, [pts], False, 0, thickness=linewidth)

				if not fill:
					tmp = np.copy(gt[:,:,cls_id])
					cv2.polylines(tmp, [pts], False, 1, thickness=linewidth)
					gt[:,:,cls_id] = tmp
	
			if fill:
				contours = []
				for coord in coords:
					for pt in coord:
						contours.append(pt)
				contours = np.array(contours).reshape((-1,1,2)).astype(np.int32)
				contours = cv2.convexHull(contours)

				tmp = np.copy(gt[:,:,cls_id])
				cv2.fillConvexPoly(tmp, points=contours, color=1)
				gt[:,:,cls_id] = tmp

		cv2.imwrite('dataset/%05d.png' % i, img * 255)
		np.save('dataset/%05d.npy' % i, gt)
		imgList.append((img, gt))
	print()

	return imgList

class FlowchartDataset(torch.utils.data.Dataset):
	def __init__(self, set, size):
		self.imgList = glob.glob('dataset/' + set + '/*.png')
		self.size = size

	def __len__(self):
		return len(self.imgList)

	def __getitem__(self, idx):
		img = cv2.imread(self.imgList[idx], 1).astype(np.float32) / 255
		gt = np.load(self.imgList[idx].replace('.png', '.npy'))

		scale = min(self.size[0] / img.shape[1], self.size[1] / img.shape[0])
		img = cv2.resize(img, None, fx=scale, fy=scale)
		img_ = np.ones((self.size[1], self.size[0], 3), dtype=np.float32)
		img_[:img.shape[0], :img.shape[1]] = img

		gt = cv2.resize(gt, None, fx=scale, fy=scale)
		gt_ = np.zeros((self.size[1], self.size[0], gt.shape[2]), dtype=np.float32)
		gt_[:gt.shape[0], :gt.shape[1]] = gt

		return 1.0 - img_.transpose((2,0,1)), gt_.transpose((2,0,1))

def computeIoU(pred, gt, thresh=0.5):
	intersection = np.bitwise_and(pred > thresh, gt > thresh).astype(np.float32)
	union = np.bitwise_or(pred > thresh, gt > thresh).astype(np.float32) + 1e-10
	return intersection.sum(axis=(0,1)) / union.sum(axis=(0,1))

if __name__ == '__main__':
#	imgList = makeImages()

	loader_train = torch.utils.data.DataLoader(FlowchartDataset(set='train', size=(256, 256)), batch_size=2, shuffle=True)
	loader_test = torch.utils.data.DataLoader(FlowchartDataset(set='test', size=(256, 256)), batch_size=2, shuffle=False)

	device = ('cuda' if torch.cuda.is_available() else 'cpu')

	model = UNet(n_channels=3, n_classes=len(class_names))
#	model = FCN(n_classes=len(class_names))
#	model = DeepLabV3(n_classes=len(class_names))
	model = model.to(device)
	summary(model, (3, 256, 256))

	optimizer = optim.Adam(model.parameters())
#	criteria = nn.BCEWithLogitsLoss()
	criteria = fl.FocalLossBCE(gamma=2.0)

	for epoch in range(10):
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
				result = model(img).cpu().numpy().transpose((0,2,3,1))

				for j in range(result.shape[0]):
					iou.append(computeIoU(result[j], answer[j]))
					cv2.imwrite('result/epoch%03d_%d_%d.png' % (epoch, i, j), 
						np.hstack((input[j] * 255, visualizeMap(result[j]), visualizeMap(answer[j]))))

			iou = np.array(iou)
			print('IoU')
			mean = np.mean(iou, axis=0)
			for i, c in enumerate(class_names):
				print('\t' + c + ': %f' % mean[i])
			print('mIoU: %f' % (np.mean(iou)))

		model.train()
