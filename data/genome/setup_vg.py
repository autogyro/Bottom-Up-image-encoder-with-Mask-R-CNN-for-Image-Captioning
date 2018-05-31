#!/usr/bin/python


''' Visual genome data analysis and preprocessing.'''

import json
import os
import operator
from pycocotools.coco import COCO
from visual_genome_python_driver import local as vg
from collections import Counter, defaultdict
import xml.etree.cElementTree as ET
from xml.dom import minidom
import sys
import numpy as np
import ast
from tqdm import tqdm
from multiprocessing import Pool, Lock, RawValue

class Counter(object):
    def __init__(self, value=0):
        # RawValue because we don't need it to create a Lock:
        self.val = RawValue('i', value)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value

# modifed by Hongyuan
dataDir = '../VGdata'
outDir = 'sb'

objects = set()
attributes = set()
relations = set()

index_to_objects = {}
index_to_attributes = {}
index_to_relations = {}


objects_to_index = {}
attributes_to_index = {}
relations_to_index = {}

with open(os.path.join(dataDir, 'scene_graphs.json')) as f:
  data = json.load(f)

with open(os.path.join("1600-400-20", 'objects_vocab.txt')) as f:
  count = 1
  for object in f.readlines():
    names = [n.lower().strip() for n in object.split(',')]
    for n in names:
      objects_to_index[n] = count
    count += 1

with open(os.path.join("1600-400-20", 'attributes_vocab.txt')) as f:
  count = 1
  for att in f.readlines():
    names = [n.lower().strip() for n in att.split(',')]
    for n in names:
      attributes_to_index[n] = count
    count += 1

with open(os.path.join("1600-400-20", 'relations_vocab.txt')) as f:
  count = 1
  for rel in f.readlines():
    names = [n.lower().strip() for n in rel.split(',')]
    for n in names:
      relations_to_index[n] = count
    count += 1

coco_objects = set()
coco_index = {}
with open('coco_category.txt') as f:
  count = 1
  for obj in f.readlines():
    names = [n.lower().strip() for n in obj.split(',')]
    coco_objects.add(names[0])
    coco_index[names[0]] = count
    count += 1

index_to_coco_objects = {}

for k, v in objects_to_index.iteritems():
  if not v in index_to_coco_objects:
    for coco_obj in coco_objects:
      if coco_obj in k:
        index_to_coco_objects[v] = coco_obj

# modifed by Hongyuan
index_to_coco_objects[objects_to_index['man']] = 'person'
index_to_coco_objects[objects_to_index['woman']] = 'person'
index_to_coco_objects[objects_to_index['tennis player']] = 'person'
index_to_coco_objects[objects_to_index['soccer player']] = 'person'
index_to_coco_objects[objects_to_index['baseball player']] = 'person'
index_to_coco_objects[objects_to_index['baseball players']] = 'person'
index_to_coco_objects[objects_to_index['players']] = 'person'
index_to_coco_objects[objects_to_index['player']] = 'person'
index_to_coco_objects[objects_to_index['people']] = 'person'
index_to_coco_objects[objects_to_index['plant']] = 'potted plant'
index_to_coco_objects[objects_to_index['plants']] = 'potted plant'

for k, v in objects_to_index.iteritems():
  if not v in index_to_coco_objects:
    continue
  print(str(v) + ' ' + k + ' ' + index_to_coco_objects[v])

# Load karpathy coco splits
coco_train = set()
with open('instances_train2014.json') as f:
  coco_data = json.load(f)
  for item in coco_data['images']:
    coco_train.add(item['id'])

coco_val = set()
with open('instances_val2014.json') as f:
  coco_data = json.load(f)
  for item in coco_data['images']:
    coco_val.add(item['id'])
print(len(coco_train))
print(len(coco_val))

annFile1='instances_train2014.json'
annFile2='instances_val2014.json'
coco1=COCO(annFile1)
coco2=COCO(annFile2)

  # Load image metadata
metadata = {}
with open(os.path.join(dataDir, 'image_data.json')) as f:
  for item in json.load(f):
    metadata[item['image_id']] = item

  # Output clean xml files, one per image
out_folder = 'xml'
if not os.path.exists(os.path.join(outDir, out_folder)):
  os.mkdir(os.path.join(outDir, out_folder))


# Set maximum values for number of object / attribute / relation classes,
# filter it further later
max_objects = 2500
max_attributes = 1000
max_relations = 500

total_area = 0
total_mask = 0

common_attributes = set(['white','black','blue','green','red','brown','yellow',
    'small','large','silver','wooden','orange','gray','grey','metal','pink','tall',
    'long','dark'])

def clean_string(string):
  string = string.lower().strip()
  if len(string) >= 1 and string[-1] == '.':
    return string[:-1].strip()
  return string

def clean_objects(string, common_attributes):
  ''' Return object and attribute lists '''
  string = clean_string(string)
  words = string.split()
  if len(words) > 1:
    prefix_words_are_adj = True
    for att in words[:-1]:
      if not att in common_attributes:
        prefix_words_are_adj = False
    if prefix_words_are_adj:
      return words[-1:],words[:-1]
    else:
      return [string],[]
  else:
    return [string],[]

def clean_attributes(string):
  ''' Return attribute list '''
  string = clean_string(string)
  if string == "black and white":
    return [string]
  else:
    return [word.lower().strip() for word in string.split(" and ")]

def clean_relations(string):
  string = clean_string(string)
  if len(string) > 0:
    return [string]
  else:
    return []

def prettify(elem):
  ''' Return a pretty-printed XML string for the Element '''
  rough_string = ET.tostring(elem, 'utf-8')
  reparsed = minidom.parseString(rough_string)
  return reparsed.toprettyxml(indent="  ")

def build_vocabs_and_xml():
  global mask_num
  global obj_num
  mask_num = 0
  obj_num = 0
  # modified by Hongyuan Lu
  for sg in tqdm(data):

    # Element and SubElement would return a handler for further referencing.
    ann = ET.Element("annotation")
    meta = metadata[sg["image_id"]]
    assert sg["image_id"] == meta["image_id"]
    url_split = meta["url"].split("/")
    outFile = url_split[-1].replace(".jpg",".xml")
    f_name = os.path.join(outDir, out_folder, outFile)
    if not os.path.exists(f_name):
        continue
    tree = ET.parse(f_name)
    objs = tree.findall('object')
    num_objs = len(objs)
    for obj in objs:
        obj_name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')
        #print(bbox.find('has_mask').text)
        has_mask = ast.literal_eval(bbox.find('has_mask').text)
        if has_mask:
            mask_num = mask_num + 1
        obj_num = obj_num + 1
  print(mask_num)
  print(obj_num)

  #pool = Pool()                         # Create a multiprocessing Pool
  #pool.map(process_image, data)


def process_image(sg):
    # Element and SubElement would return a handler for further referencing.
    ann = ET.Element("annotation")
    meta = metadata[sg["image_id"]]
    assert sg["image_id"] == meta["image_id"]
    url_split = meta["url"].split("/")

    outFile = url_split[-1].replace(".jpg",".xml")
    if os.path.exists(os.path.join(outDir, out_folder, outFile)):
        #print('skipping' + outFile)
        return
    # https://cs.stanford.edu/people/rak248/VG_100K/2.jpg
    ET.SubElement(ann, "folder").text = url_split[-2]
    ET.SubElement(ann, "filename").text = url_split[-1]

    source = ET.SubElement(ann, "source")
    ET.SubElement(source, "database").text = "Visual Genome Version 1.2"
    ET.SubElement(source, "image_id").text = str(meta["image_id"])
    ET.SubElement(source, "coco_id").text = str(meta["coco_id"])



    ET.SubElement(source, "flickr_id").text = str(meta["flickr_id"])

    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(meta["width"])
    ET.SubElement(size, "height").text = str(meta["height"])
    ET.SubElement(size, "depth").text = "3"

    ET.SubElement(ann, "segmented").text = "0"

    object_set = set()

    # modifed by Hongyuan
    for obj in sg['objects']:
      has_mask = True
      coco_i = 0
      thread_safe_counter1.increment()
      o,a = clean_objects(obj['names'][0], common_attributes)
      if o[0] in objects_to_index:
        ob = ET.SubElement(ann, "object")
        ET.SubElement(ob, "name").text = o[0]
        ET.SubElement(ob, "object_id").text = str(obj["object_id"])
        object_set.add(obj["object_id"])
        new_mask = np.zeros((28, 28), dtype=int)
        coco = coco1
        if meta["coco_id"] in coco_val:
          coco = coco2
        if (meta["coco_id"] in coco_val or meta["coco_id"] in coco_train) and objects_to_index[o[0]] in index_to_coco_objects.keys():
          coco_object = index_to_coco_objects[objects_to_index[o[0]]]

          catIds = coco.getCatIds(catNms=[coco_object]);
          annIds = coco.getAnnIds(imgIds=meta["coco_id"], iscrowd=None)
          anns = coco.loadAnns(annIds)
          imgIds = coco.getImgIds(imgIds=[meta["coco_id"]]);
          img = coco.loadImgs(imgIds)[0]
          img_width = img['width']
          img_height = img['height']
          height_ratio = meta["height"] * 1.0 / img_height
          width_ratio = meta["width"] * 1.0 / img_width
          mask = np.zeros((img_height, img_width), dtype=int)

          x1 = obj["x"] / width_ratio
          x2 = (obj["x"] + obj["w"]) / width_ratio
          y1 = obj["y"] / height_ratio
          y2 = (obj["y"] + obj["h"]) / height_ratio
          if x2 < x1 or y2 < y1:
            has_mask = False
          if has_mask:
            has_mask = False
            for annn in anns:
              decoded_mask = coco.annToMask(annn)
              coco_obj_name = coco.loadCats(ids=[annn['category_id']])
              (x, y) = decoded_mask.shape
              c1 = 0
              c2 = 0
              for i in range(x):
                for j in range(y):
                  if decoded_mask[i][j] == 1:
                    c1 = c1 + 1
                    if j >= x1 and j <= x2 and i >= y1 and i <= y2:
                      c2 = c2 + 1
              #print(c1)
              #print(c2)
              if (not c1 == 0) and c2 * 1.0 / c1 >= 0.9:
                has_mask = True
                coco_i = coco_index[coco_obj_name[0]['name']]
                mask = np.bitwise_or(mask, decoded_mask.astype(int))
          #object_id = objects_to_index
          if has_mask:
            x_width = (x2 - x1) / 27.0
            y_width = (y2 - y1) / 27.0
            for i in range (28):
              for j in range(28):
                new_mask[i][j]=mask[int(y1 + y_width * i) - 1][int(x1 + x_width * j) - 1]

            #new_mask_str = np.array2string(new_mask, precision=1, separator=',', suppress_small=False)
            #print(np.asarray(ast.literal_eval(new_mask_str)).shape)
        else:
          has_mask = False

        ET.SubElement(ob, "difficult").text = "0"
        bbox = ET.SubElement(ob, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(obj["x"])
        ET.SubElement(bbox, "ymin").text = str(obj["y"])
        ET.SubElement(bbox, "xmax").text = str(obj["x"] + obj["w"])
        ET.SubElement(bbox, "ymax").text = str(obj["y"] + obj["h"])
        ET.SubElement(bbox, "mask").text = np.array2string(new_mask, precision=1, separator=',', suppress_small=False)
        ET.SubElement(bbox, "has_mask").text = str(has_mask)
        ET.SubElement(bbox, "coco_category").text = str(coco_i)
        attribute_set = set()
        for attribute_name in a:
          if attribute_name in attributes_to_index:
            attribute_set.add(attribute_name)
        for attr in sg['attributes']:
          if attr["attribute"]["object_id"] == obj["object_id"]:
            try:
              for ix in attr['attribute']['attributes']:
                for clean_attribute in clean_attributes(ix):
                  if clean_attribute in attributes:
                    attribute_set.add(clean_attribute)
            except:
              pass
        # for loop
        for attribute_name in attribute_set:
          ET.SubElement(ob, "attribute").text = attribute_name

    for rel in sg['relationships']:
      predicate = clean_string(rel["predicate"])
      if rel["subject_id"] in object_set and rel["object_id"] in object_set:
        if predicate in relations_to_index:
          re = ET.SubElement(ann, "relation")
          ET.SubElement(re, "subject_id").text = str(rel["subject_id"])
          ET.SubElement(re, "object_id").text = str(rel["object_id"])
          ET.SubElement(re, "predicate").text = predicate

    outFile = url_split[-1].replace(".jpg",".xml")
    tree = ET.ElementTree(ann)
    if len(tree.findall('object')) > 0:
      tree.write(os.path.join(outDir, out_folder, outFile))
    print("done" + outFile)

if __name__ == "__main__":
  print("here1")
  # First, use visual genome library to merge attributes and scene graphs
  #vg.AddAttrsToSceneGraphs(dataDir=dataDir)
  # Next, build xml files
  print("here2")
  build_vocabs_and_xml()
