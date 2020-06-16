import numpy as np
import math
import urllib.request
import io
import cv2

def deg2num(lat_deg, lon_deg, zoom):
	lat_rad = math.radians(lat_deg)
	n = 2.0 ** zoom
	xtile = ((lon_deg + 180.0) / 360.0 * n)
	ytile = ((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
	return (int(xtile), int(ytile), xtile - int(xtile), ytile - int(ytile))

def num2deg(xtile, ytile, zoom):
	n = 2.0 ** zoom
	lon_deg = xtile / n * 360.0 - 180.0
	lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
	lat_deg = math.degrees(lat_rad)
	return (lat_deg, lon_deg)

def getImageCluster(lat1, lon1, lat2, lon2, zoom):
	bs = 256
	headers = {
		'User-Agent': 'C4ISR 0.1 [contact@adress.com]'
	}
	smurl = r"https://a.tile.openstreetmap.org/{0}/{1}/{2}.png"
	xmin, ymax, xmin_, ymax_ = deg2num(lat1, lon1, zoom)
	xmax, ymin, xmax_, ymin_ = deg2num(lat2, lon2, zoom)
	xmax += 1
	ymax += 1

	cluster = np.zeros(((ymax-ymin+1)*bs, (xmax-xmin+1)*bs, 3), dtype=np.uint8)
	for xtile in range(xmin, xmax+1):
		for ytile in range(ymin, ymax+1):
			imgurl = smurl.format(zoom, xtile, ytile)
			try:
				with urllib.request.urlopen(urllib.request.Request(imgurl, headers=headers)) as res:
					tile = cv2.imdecode(np.frombuffer(res.read(), dtype=np.uint8), -1)
					cluster[(ytile-ymin)*bs:(ytile-ymin+1)*bs,(xtile-xmin)*bs:(xtile-xmin+1)*bs] = tile
			except urllib.error.HTTPError as e:
				if e.code >= 400:
					print(e)
				else:
					raise e

	return cluster[int((1-ymin_)*bs):-int((1-ymax_)*bs), int(xmin_*bs):-int(xmax_*bs)]

if __name__ == '__main__':
	a = getImageCluster(33.60487679443204, 130.20721019245002, 33.59553966764103, 130.23941063622806, 13)
	cv2.imshow('', a)
	cv2.waitKey()
