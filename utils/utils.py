import torch
import cv2
import numpy as np
import glob
import random
from . import flowchart
from . import osm

np.random.seed(114514)

class_names = ['process', 'decision', 'terminator', 'data', 'connection', 'arrow', 'text']
fill_flag = [True, True, True, True, True, False, True]
color = [(0,0,255), (0,255,0), (255,255,0), (0,255,255), (255,0,255), (255,0,0), (255,255,255)]
osm_cands = glob.glob('dataset/osm_cache/*.png')

def imshow(img, title='', scale=0.25):
	cv2.imshow(title, cv2.resize(img, None, fx=scale, fy=scale))

def visualizeMap(map):
	img = np.zeros((map.shape[0], map.shape[1], 3), dtype=np.uint8)
	for i in range(map.shape[2]):
		img[map[:,:,i] > 0] = color[i]
	return img

def visualizeInst(img, inst, inst_class):
	if len(img.shape) == 3 and img.shape[2] == 3:
		img = img.copy()
	else:
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

	for i in range(inst.shape[2]):
		cnt = np.array(np.where(inst[:,:,i] > 0)).transpose((1,0))[:,::-1]
		if cnt.shape[0] == 0:
			continue
		x,y,w,h = cv2.boundingRect(cnt)
		img = cv2.rectangle(img, (x, y), (x+w, y+h), color[inst_class[i]], 3)
	return img

def encodeMap(map):
	img = np.zeros((map.shape[0], map.shape[1]), dtype=np.uint8)
	for i in range(map.shape[2]):
		img[map[:,:,i] > 0] += 2 ** i
	return img

def decodeMap(enc_map):
	map = np.zeros((enc_map.shape[0], enc_map.shape[1], len(class_names)), dtype=np.float32)
	for i in range(map.shape[2] - 1, -1, -1):
		map[:,:,i][enc_map >= 2 ** i] = 1
		enc_map[enc_map >= 2 ** i] -= 2 ** i
	return map

class FlowchartDataset(torch.utils.data.Dataset):
	def __init__(self, set, size, aug=False, osm_alpha=0.0, max_inst_num=50):
		self.imgList = glob.glob('dataset/' + set + '/img/*.png')
		self.size = size
		self.osm_alpha = osm_alpha
		self.aug = aug
		self.max_inst_num = max_inst_num

	def __len__(self):
		return len(self.imgList)

	def __getitem__(self, idx):
		img = cv2.imread(self.imgList[idx], 1).astype(np.float32) / 255
		gt = decodeMap(cv2.imread(self.imgList[idx].replace('img', 'gt'), 0))
		inst_arr = np.load(self.imgList[idx].replace('img', 'inst').replace('.png', '.npz'))
		inst, inst_class = inst_arr['arr_0'], inst_arr['arr_1']

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

		inst_ = np.zeros((self.size[1], self.size[0], self.max_inst_num), dtype=np.float32)
		for i in range(inst.shape[2]):
			tmp =  cv2.resize(inst[:,:,i], None, fx=scale, fy=scale)
			inst_[:tmp.shape[0], :tmp.shape[1], i] = tmp

		if self.aug:
			gt_ = encodeMap(gt_)
			img_ = img_ * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.1, 0.1)

			if np.random.rand() >= 0.5:
				img_ = np.flipud(img_).copy()
				gt_ = np.flipud(gt_).copy()
				inst_ = np.flipud(inst_).copy()

			if np.random.rand() >= 0.5:
				img_ = np.fliplr(img_).copy()
				gt_ = np.fliplr(gt_).copy()
				inst_ = np.fliplr(inst_).copy()

			angle = np.random.uniform(-30, 30)
			scale = np.random.uniform(0.8, 1.0)
			trans = cv2.getRotationMatrix2D((img_.shape[1]/2, img_.shape[0]/2), angle, scale)
			img_ = cv2.warpAffine(img_, trans, (img_.shape[1],img_.shape[0]))
			gt_ = cv2.warpAffine(gt_, trans, (gt_.shape[1],gt_.shape[0]), flags=cv2.INTER_NEAREST)
			for i in range(inst_.shape[2]):
				inst_[:,:,i] = cv2.warpAffine(inst_[:,:,i], trans, (inst_.shape[1],inst_.shape[0]), flags=cv2.INTER_NEAREST)

			gt_ = decodeMap(gt_)

		inst_class_ = np.ones((self.max_inst_num,), dtype=np.int32) * -1
		inst_class_[:inst_class.shape[0]] = inst_class

		return img_.transpose((2,0,1)), gt_.transpose((2,0,1)), inst_.transpose((2,0,1)), inst_class_

def computeIoU(pred, gt, thresh=0.5):
	intersection = np.bitwise_and(pred > thresh, gt > thresh).astype(np.float32)
	union = np.bitwise_or(pred > thresh, gt > thresh).astype(np.float32) + 1e-10
	return intersection.sum(axis=(0,1)) / union.sum(axis=(0,1))

def makeImages(linewidth=5, max_inst_num=50):
	dataset = flowchart.load()

#	imgList = []

	for i, data in enumerate(dataset):
		print('\r%d/%d' % (i, len(dataset)), end='')

		img = np.ones((int(data['size'][1]), int(data['size'][0])), dtype=np.float32)
		gt = np.zeros((int(data['size'][1]), int(data['size'][0]), len(class_names)), dtype=np.uint8)
		inst = np.zeros((int(data['size'][1]), int(data['size'][0]), max_inst_num), dtype=np.uint8)
		inst_class = []

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

					tmp = np.copy(inst[:,:,c])
					cv2.polylines(tmp, [pts], False, 1, thickness=linewidth)
					inst[:,:,c] = tmp
	
			if fill:
				contours = []
				for coord in coords:
					for pt in coord:
						contours.append(pt)
				contours = np.array(contours).reshape((-1,1,2)).astype(np.int32)

				if class_names[cls_id] == 'text':
					rect = cv2.minAreaRect(contours)
					box = cv2.boxPoints(rect)
					box = np.int0(box)
					contours = box
				else:
					contours = cv2.convexHull(contours)

				tmp = np.copy(gt[:,:,cls_id])
				cv2.fillConvexPoly(tmp, points=contours, color=1)
				gt[:,:,cls_id] = tmp

				tmp = np.copy(inst[:,:,c])
				cv2.fillConvexPoly(tmp, points=contours, color=1)
				inst[:,:,c] = tmp

			inst_class.append(cls_id)
		inst_class = np.array(inst_class)

		cv2.imwrite('dataset/img/%05d.png' % i, img * 255)
		cv2.imwrite('dataset/gt/%05d.png' % i, encodeMap(gt))
		np.savez_compressed('dataset/inst/%05d.npz' % i, inst[:,:,:len(inst_class)], inst_class)

#		imgList.append((img, gt, inst))
	print()

#	return imgList

if __name__ == '__main__':
	makeImages()
