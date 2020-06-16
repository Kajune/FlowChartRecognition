import numpy as np
import cv2
import glob
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

import flowchart
import focalLoss as fl
from model.unet import UNet, UNetPP
from model.fcn import *
import osm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='UNet', choices=['FCN', 'DeepLab', 'UNet', 'UNetPP'])
args = parser.parse_args()

class_names = ['process', 'decision', 'terminator', 'data', 'connection', 'arrow', 'text']
fill_flag = [True, True, True, True, True, False, False]
color = [(0,0,255), (0,255,0), (255,255,0), (0,255,255), (255,0,255), (255,0,0), (255,255,255)]

np.random.seed(114514)

osm_cands = glob.glob('dataset/osm_cache/*.png')

def imshow(img, title='', scale=0.25):
	cv2.imshow(title, cv2.resize(img, None, fx=scale, fy=scale))

def visualizeMap(map):
	img = np.zeros((map.shape[0], map.shape[1], 3), dtype=np.uint8)
	for i in range(map.shape[2]):
		img[map[:,:,i] > 0] = color[i]
	return img

def encodeMap(map):
	img = np.zeros((map.shape[0], map.shape[1]), dtype=np.uint8)
	for i in range(map.shape[2]):
		img[map[:,:,i] > 0] += 2 ** (i + 1)
	return img

def decodeMap(enc_map):
	map = np.zeros((enc_map.shape[0], enc_map.shape[1], len(class_names)), dtype=np.float32)
	for i in range(map.shape[2] - 1, -1, -1):
		map[:,:,i][enc_map >= 2 ** (i + 1)] = 1
		enc_map[enc_map >= 2 ** (i + 1)] -= 2 ** (i + 1)
	return map

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

		cv2.imwrite('dataset/img/%05d.png' % i, img * 255)
		cv2.imwrite('dataset/gt/%05d.png' % i, encodeMap(gt))
		imgList.append((img, gt))
	print()

	return imgList

class FlowchartDataset(torch.utils.data.Dataset):
	def __init__(self, set, size, aug=False, osm_alpha=0.0):
		self.imgList = glob.glob('dataset/' + set + '/img/*.png')
		self.size = size
		self.osm_alpha = osm_alpha
		self.aug = aug

	def __len__(self):
		return len(self.imgList)

	def __getitem__(self, idx):
		img = cv2.imread(self.imgList[idx], 1).astype(np.float32) / 255
		gt = decodeMap(cv2.imread(self.imgList[idx].replace('img', 'gt'), 0))

		scale = min(self.size[0] / img.shape[1], self.size[1] / img.shape[0])
		img = cv2.resize(img, None, fx=scale, fy=scale)
		img_ = np.ones((self.size[1], self.size[0], 3), dtype=np.float32)
		img_[:img.shape[0], :img.shape[1]] = img

		if self.osm_alpha == 'random' or self.osm_alpha > 0:
			"""
			lat, lon, d, zoom = random.choice(osm_cands)

			if os.path.exists('dataset/osm_cache/%f_%f_%f_%d.png' % (lat, lon, d, zoom)):
				mapImg = cv2.imread('dataset/osm_cache/%f_%f_%f_%d.png' % (lat, lon, d, zoom))
			else:
				mapImg = osm.getImageCluster(lat, lon, lat + d, lon + d, zoom)
				cv2.imwrite('dataset/osm_cache/%f_%f_%f_%d.png' % (lat, lon, d, zoom), mapImg)
			"""
			mapImg = cv2.imread(random.choice(osm_cands))
			mapImg = cv2.resize(mapImg, (self.size[1], self.size[0])).astype(np.float32) / 255
			mapImg *= np.random.uniform(0.0, 1.0) if self.osm_alpha == 'random' else self.osm_alpha
			mapImg += 1.0 - np.max(mapImg)
			img_ = mapImg * img_

		gt = cv2.resize(gt, None, fx=scale, fy=scale)
		gt_ = np.zeros((self.size[1], self.size[0], gt.shape[2]), dtype=np.float32)
		gt_[:gt.shape[0], :gt.shape[1]] = gt

		if self.aug:
			gt_ = encodeMap(gt_)
			img_ = img_ * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.1, 0.1)

			if np.random.rand() >= 0.5:
				img_ = np.flipud(img_).copy()
				gt_ = np.flipud(gt_).copy()

			if np.random.rand() >= 0.5:
				img_ = np.fliplr(img_).copy()
				gt_ = np.fliplr(gt_).copy()

			angle = np.random.uniform(-30, 30)
			scale = np.random.uniform(0.8, 1.0)
			trans = cv2.getRotationMatrix2D((img_.shape[1]/2, img_.shape[0]/2), angle, scale)
			img_ = cv2.warpAffine(img_, trans, (img_.shape[1],img_.shape[0]))
			gt_ = cv2.warpAffine(gt_, trans, (gt_.shape[1],gt_.shape[0]), flags=cv2.INTER_NEAREST)

			gt_ = decodeMap(gt_)

		return img_.transpose((2,0,1)), gt_.transpose((2,0,1))

def computeIoU(pred, gt, thresh=0.5):
	intersection = np.bitwise_and(pred > thresh, gt > thresh).astype(np.float32)
	union = np.bitwise_or(pred > thresh, gt > thresh).astype(np.float32) + 1e-10
	return intersection.sum(axis=(0,1)) / union.sum(axis=(0,1))

if __name__ == '__main__':
#	imgList = makeImages()

	size=(384,384)

	loader_train = torch.utils.data.DataLoader(FlowchartDataset(set='train', size=size, osm_alpha='random', aug=True), 
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
