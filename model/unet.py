""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
	"""(convolution => [BN] => ReLU) * 2"""

	def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, dilation=1):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=dilation + kernel_size//2 - 1, dilation=dilation),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=dilation + kernel_size//2 - 1, dilation=dilation),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)


class Down(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels, dilation=1):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels, dilation=dilation)
		)

	def forward(self, x):
		return self.maxpool_conv(x)


class Up(nn.Module):
	"""Upscaling then double conv"""

	def __init__(self, in_channels, out_channels, bilinear=True, dilation=1):
		super().__init__()

		# if bilinear, use the normal convolutions to reduce the number of channels
		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		else:
			self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

		self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dilation=dilation)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		# input is CHW
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		# if you have padding issues, see
		# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
		# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)

class MultiUp(nn.Module):
	"""Upscaling then double conv"""

	def __init__(self, in_channels, out_channels, bilinear=True, dilation=1):
		super().__init__()

		# if bilinear, use the normal convolutions to reduce the number of channels
		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		else:
			self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

		self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dilation=dilation)

	def forward(self, x1, x2_list):
		x1 = self.up(x1)
		# input is CHW
		diffY = x2_list[0].size()[2] - x1.size()[2]
		diffX = x2_list[0].size()[3] - x1.size()[3]

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		# if you have padding issues, see
		# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
		# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
		cat = []
		for x in x2_list:
			cat.append(x)
		cat.append(x1)
		x = torch.cat(cat, dim=1)
		return self.conv(x)

class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(OutConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x):
		return self.conv(x)

class UNet(nn.Module):
	def __init__(self, n_channels, n_classes, depth=4, base_channels=32, dilation=2, bilinear=True):
		super().__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear
		self.depth = depth

		bc = base_channels

		self.inc = DoubleConv(n_channels, bc)		
		self.outc = OutConv(bc, n_classes)

		downlayer = []
		uplayer = []

		for i in range(self.depth-1):
			downlayer.append(Down(bc*2**i, bc*2**(i+1), dilation=dilation))
			uplayer.append(Up(bc*2**(self.depth-i), bc*2**(self.depth-i-2), bilinear, dilation=dilation))
		downlayer.append(Down(bc*2**(self.depth-1), bc*2**(self.depth-1), dilation=dilation))
		uplayer.append(Up(bc*2, bc, bilinear, dilation=dilation))

		self.downlayer = nn.ModuleList(downlayer)
		self.uplayer = nn.ModuleList(uplayer)

	def forward(self, x):
		x = self.inc(x)

		mid = []
		for i in range(len(self.downlayer)):
			mid.append(x)
			x = self.downlayer[i](x)

		for i in range(len(self.uplayer)):
			x = self.uplayer[i](x, mid[-i-1])

		return self.outc(x)

	def predict(self, x):
		return self.forward(x)

class UNetPP(nn.Module):
	def __init__(self, n_channels, n_classes, depth=4, base_channels=16, bilinear=True):
		super().__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear
		self.depth = depth

		bc = base_channels

		self.inc = DoubleConv(n_channels, bc)		
		self.outc = OutConv(bc, n_classes)

		downlayer = []
		for i in range(self.depth):
			downlayer.append(Down(bc*2**i, bc*2**(i+1), dilation=1))
		self.downlayer = nn.ModuleList(downlayer)

		uplayer = []
		for i in range(self.depth):
			sublayer = []
			for j in range(i + 1):
				sublayer.append(MultiUp(bc*2**(self.depth-i-1)*(j+1+2), bc*2**(self.depth-i-1), bilinear, dilation=1))
			uplayer.append(nn.ModuleList(sublayer))

		self.uplayer = nn.ModuleList(uplayer)

	def forward(self, x, mean=False, prune=None):
		x = self.inc(x)

		mid = []
		for i in range(len(self.downlayer) if prune is None else prune):
			mid.append(x)
			x = self.downlayer[i](x)

		last_output = [x]
		for i in range(0 if prune is None else len(self.uplayer) - prune, len(self.uplayer)):
			mid_idx = (-i-1) if prune is None else (len(self.uplayer)-i-1)
			sub_output = [mid[mid_idx]]
			for j in range(len(self.uplayer[i]) if prune is None else (prune - len(self.uplayer) + i + 1)):
				x = self.uplayer[i][j](last_output[j], sub_output)
				sub_output.append(x)
			last_output = sub_output

		if mean:
			out_mean = []
			for o in last_output:
				out_mean.append(self.outc(o))
			return torch.mean(torch.stack(out_mean), dim=0)
		else:
			return self.outc(last_output[-1])

	def predict(self, x, prune=None):
		return self.forward(x, False, prune)

class SOLO(nn.Module):
	def __init__(self, model, n_classes, in_channels=256, n_grid=12):
		super().__init__()

		self.model = model
		self.in_channels = in_channels
		self.n_classes = n_classes
		self.n_grid = n_grid

		self.category_output_layer = DoubleConv(in_channels, n_classes, kernel_size=1)

		self.mask_output_layer = [DoubleConv(in_channels + 2, in_channels)]
		for i in range(7):
			self.mask_output_layer.append(DoubleConv(in_channels, in_channels))
		self.mask_output_layer.append(DoubleConv(in_channels, n_grid ** 2, kernel_size=1))
		self.mask_output_layer = nn.Sequential(*self.mask_output_layer)

	def forward(self, x):
		feature = self.model(x)
		category = self.category_output_layer(feature)

		coord_x = torch.linspace(-1, 1, steps=feature.shape[2]).to(feature) \
			.view(1,1,feature.shape[2],1).repeat(feature.shape[0],1,1,feature.shape[3])
		coord_y = torch.linspace(-1, 1, steps=feature.shape[3]).to(feature) \
			.view(1,1,1,feature.shape[3]).repeat(feature.shape[0],1,feature.shape[2],1)
		mask = self.mask_output_layer(torch.stack([feature, coord_x, coord_y], dim=1))

		return category, mask

	def predict(self, x, thresh=0.1, top_n=500):
		category, mask = self.forward(x)
		cat_grid = F.interpolate(category, size=(cat_grid.shape[2]//self.n_grid, cat_grid.shape[3]//self.n_grid), mode='bilinear')
		topk = torch.topk(cat_grid.view(cat_grid.shape[0], cat_grid.shape[1], -1), top_n, largest=True)
