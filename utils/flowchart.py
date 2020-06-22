import xml.etree.ElementTree as ET
import glob
import numpy as np

def load():
	inkmlList = glob.glob('dataset/FCinkML/*.inkml')
	inkmlList = sorted(inkmlList)

	imageList = []

	for inkml in inkmlList:
		tree = ET.parse(inkml)
		root = tree.getroot()

		doc_ns = "{http://www.w3.org/2003/InkML}"

		annot_ns = "{LUNAM/IRCCyN/FlowchartML}"
		annots = root.find(doc_ns + 'annotationXML').find(annot_ns + 'flowchart')
		annotDict = {}

		for annots_tag in annots:
			id = annots_tag.attrib['{http://www.w3.org/XML/1998/namespace}id']
			annotDict[id] = annots_tag

			for subAnnot in annots_tag:
				id = subAnnot.attrib['{http://www.w3.org/XML/1998/namespace}id']
				annotDict[id] = subAnnot

		traces = root.findall(doc_ns + 'trace')
		traceDict = {}

		for trace_tag in traces:
			id = int(trace_tag.get('id'))
			traceDict[id] = trace_tag

		traceGroup = root.find(doc_ns + 'traceGroup')

		coord_array_list = []
		annotation_list = []

		minCoord = None
		maxCoord = None

		for tg in traceGroup.findall(doc_ns + 'traceGroup'):
			class_name = tg.find(doc_ns + 'annotation').text
			annot = tg.find(doc_ns + 'annotationXML')
			if annot is not None:
				annot_id = annot.attrib['href']
				annot = annotDict[annot_id]
				if class_name == 'text':
					annot = annot.text
				else:
					annot = annot.attrib

			annotation_list.append((class_name, annot))

			coord_array = []
			for trace in tg.findall(doc_ns + 'traceView'):
				coord_array_sub = []
				for coords in traceDict[int(trace.attrib['traceDataRef'])].text.replace('\n', '').split(', '):
					coords = coords.split(' ')
					coord_array_sub.append(coords)

				coord_array_sub = np.float32(coord_array_sub)
				coord_array.append(coord_array_sub)

				min = np.min(coord_array_sub, axis=0)
				max = np.max(coord_array_sub, axis=0)
				minCoord = np.min(np.array([minCoord, min]), axis=0) if minCoord is not None else min
				maxCoord = np.max(np.array([maxCoord, max]), axis=0) if maxCoord is not None else max

			coord_array_list.append(coord_array)

		margin = (maxCoord - minCoord) * 0.1
		minCoord -= margin
		maxCoord += margin
		for coord_array in coord_array_list:
			for coord_array_sub in coord_array:
				coord_array_sub -= minCoord
		maxCoord -= minCoord

		imageList.append({'annot' : annotation_list, 'size' : maxCoord, 'coords' : coord_array_list})

	return imageList

