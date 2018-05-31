#!/usr/bin/env python


"""Generate bottom-up attention features as a tsv file. Can use multiple gpus, each produces a
   separate tsv file that can be merged later (e.g. by using merge_tsv function).
   Modify the load_image_ids script as necessary for your data location. """


# Example:
# ./tools/generate_tsv.py --gpu 0,1,2,3,4,5,6,7 --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end/test.prototxt --out test2014_resnet101_faster_rcnn_genome.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --split coco_test2014


import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import caffe
import argparse
import pprint
import time, os, sys
import base64
import numpy as np
import cv2
import csv
from multiprocessing import Process
import random
import json
from tqdm import tqdm
import os
import sys

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']

# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36.
MIN_BOXES = 10
MAX_BOXES = 100

def load_image_ids(split_name):
    split = []
    with open('karpathy_test_images.txt') as f:
      for line in f:
        image_id = int(line.split()[-1])
        file_name = line.split()[-2]
        filepath = os.path.join('coco', file_name)
        maskedfilepath = os.path.join('mask', os.path.basename(file_name))
        print(maskedfilepath)
        split.append((maskedfilepath,filepath,image_id))

    return split


def get_detections_from_im(net, im_file, image_id, conf_thresh=0.2):

    im = cv2.imread(im_file)
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    pool5 = net.blobs['pool5_flat'].data

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1,cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

    return {
        'image_id': image_id,
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes' : len(keep_boxes),
        'boxes':  base64.b64encode(cls_boxes[keep_boxes]),
        'features':  base64.b64encode(pool5[keep_boxes])
    }


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to use',
                        default=None, type=str)
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--split', dest='data_split',
                        help='dataset to use',
                        default='karpathy_train', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def mask_generation(net, im_file, item, conf_thresh=0.2):

    im = cv2.imread(im_file)
    boxes = item['boxes']

    scores, boxes, attr_scores, rel_scores = im_detect(net, im, boxes, force_boxes=True)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    features2 = net.blobs['pool5_flat'].data

    item['features'] = base64.b64encode((np.concatenate((item['features'], features2), axis=1)))
    item['boxes'] = base64.b64encode(item['boxes'])

    return item

def generate_tsv(found_ids, gpu_id, prototxt, weights, image_ids, outfile):
    # First check if file exists, and if it is complete
    wanted_ids = set([int(image_id[2]) for image_id in image_ids])
    missing = wanted_ids - found_ids

    if len(missing) > 0:
        caffe.set_mode_gpu()
        caffe.set_device(0)
        with open(outfile, 'ab') as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)
            _t = {'misc' : Timer()}
            count = 0
            items={}
            for maskedfilepath,im_file,image_id in tqdm(image_ids):
                if int(image_id) in missing:
                    _t['misc'].tic()
                    net = caffe.Net(prototxt, caffe.TEST, weights=weights)
                    item = get_detections_from_im(net, im_file, image_id)
                    items[image_id]=item
                    _t['misc'].toc()
                    if (count % 100) == 0:
                        print ('projected finish: {:.2f} hours)' +  str(_t['misc'].average_time*(len(missing)-count)/3600))
                    count = count + 1

            del net
            net2 = caffe.Net("models/vg/ResNet-101/faster_rcnn_end2end_final/test_gt.prototxt", caffe.TEST, weights=weights)

            for maskedfilepath,im_file,image_id in tqdm(image_ids):
                if int(image_id) in missing:
                    print('Processing' + maskedfilepath)
                    _t['misc'].tic()
                    net2 = caffe.Net("models/vg/ResNet-101/faster_rcnn_end2end_final/test_gt.prototxt", caffe.TEST, weights=weights)
                    item = items[image_id]
                    item = mask_generation(net2, maskedfilepath, item)
                    writer.writerow(item)
                    _t['misc'].toc()
                    if (count % 100) == 0:
                        print ('projected finish: {:.2f} hours)' +  str(_t['misc'].average_time*(len(missing)-count)/3600))
                    count = count + 1
    return len(wanted_ids) - len(missing)

def generate_tsv_mask_only(found_ids, gpu_id, prototxt, weights, image_ids, outfile):
    # First check if file exists, and if it is complete
    wanted_ids = set([int(image_id[2]) for image_id in image_ids])
    missing = wanted_ids - found_ids

    if len(missing) > 0:
        caffe.set_mode_gpu()
        caffe.set_device(0)
        with open(outfile, 'ab') as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)
            _t = {'misc' : Timer()}
            count = 0

            for maskedfilepath,im_file,image_id in tqdm(image_ids):
                if int(image_id) in missing:
                    _t['misc'].tic()
                    net = caffe.Net(prototxt, caffe.TEST, weights=weights)
                    item = get_detections_from_im(net, maskedfilepath, image_id)
                    writer.writerow(item)
                    _t['misc'].toc()
                    if (count % 100) == 0:
                        print ('projected finish: {:.2f} hours)' +  str(_t['misc'].average_time*(len(missing)-count)/3600))
                    count = count + 1

    return len(wanted_ids) - len(missing)


def merge_tsvs():
    test = ['/work/data/tsv/test2015/resnet101_faster_rcnn_final_test.tsv.%d' % i for i in range(8)]

    outfile = '/work/data/tsv/merged.tsv'
    with open(outfile, 'ab') as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)

        for infile in test:
            with open(infile) as tsv_in_file:
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
                for item in reader:
                    try:
                      writer.writerow(item)
                    except Exception as e:
                      print e


def separate_tsvs():
    infile = 'karpathy_train_resnet101_faster_rcnn_genome.tsv'
    outfile1 = 'karpathy_train_resnet101_faster_rcnn_genome.tsv.0'
    outfile2 = 'karpathy_train_resnet101_faster_rcnn_genome.tsv.1'
    outfile3 = 'karpathy_train_resnet101_faster_rcnn_genome.tsv.2'

    with open(infile) as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        with open(outfile1, 'ab') as tsvfile1:
            with open(outfile2, 'ab') as tsvfile2:
                with open(outfile3, 'ab') as tsvfile3:
                    total = 120000
                    clip1 = total / 3.0
                    clip2 = clip1 * 2
                    writer1 = csv.DictWriter(tsvfile1, delimiter = '\t', fieldnames = FIELDNAMES)
                    writer2 = csv.DictWriter(tsvfile2, delimiter = '\t', fieldnames = FIELDNAMES)
                    writer3 = csv.DictWriter(tsvfile3, delimiter = '\t', fieldnames = FIELDNAMES)
                    print(total)
                    print(clip1)
                    print(clip2)
                    for i, item in tqdm(enumerate(reader)):
                        writer = writer1
                        if i > clip1:
                            writer = writer2
                        if i > clip2:
                            writer = writer3
                        try:
                            writer.writerow(hongyuan(item))
                        except Exception as e:
                            print e

def hongyuan(item):

    item['features'] = np.frombuffer(base64.decodestring(item['features']), dtype=np.float32).reshape((int(item['num_boxes']),-1))
    features1 = item['features'][: ,:2048]
    features2 = item['features'][: ,2048:]
    item['features'] = base64.b64encode(0.1 * features1 + 0.9 * features2)

    return item

def val_2048():
    infile = 'karpathy_val_resnet101_faster_rcnn_genome.tsv'
    outfile = 'karpathy_val_resnet101_faster_rcnn_genome.tsv.0'

    with open(infile) as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        with open(outfile, 'ab') as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)

            for i, item in tqdm(enumerate(reader)):

               try:
                   writer.writerow(hongyuan(item))
               except Exception as e:
                   print e

def test_2048():
    infile = 'karpathy_test_resnet101_faster_rcnn_genome.tsv'
    outfile = 'karpathy_test_resnet101_faster_rcnn_genome.tsv.0'

    with open(infile) as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        with open(outfile, 'ab') as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)

            for i, item in tqdm(enumerate(reader)):

               try:
                   writer.writerow(hongyuan(item))
               except Exception as e:
                   print e

if __name__ == '__main__':
    args = parse_args()
    #separate_tsvs()
    #val_2048()
    #test_2048()
    #sys.exit(0)
    os.environ['GLOG_minloglevel'] = '3' # suprress Caffe verbose prints
    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    gpu_id = args.gpu_id
    gpu_list = gpu_id.split(',')
    gpus = [int(i) for i in gpu_list]

    print('Using config:')
    pprint.pprint(cfg)
    assert cfg.TEST.HAS_RPN

    image_ids = load_image_ids(args.data_split)

    # Split image ids between gpus
    image_ids = [image_ids[i::1] for i in range(1)]


    caffe.init_log()
    caffe.log('Using devices %s' % str(gpus))
    procs = []
    t = 0

    found_ids = set()
    outfile = '%s.%d' % (args.outfile, 0)
    if os.path.exists(outfile):
        with open(outfile) as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames = FIELDNAMES)
            for item in tqdm(reader):
                found_ids.add(int(item['image_id']))

   # outfile = '%s.%d' % (args.outfile, 1)
   # if os.path.exists(outfile):
   #     with open(outfile) as tsvfile:
   #         reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames = FIELDNAMES)
   #         for item in tqdm(reader):
   #             found_ids.add(int(item['image_id']))
    processed = 0

    for i,image_id in enumerate(image_ids):
        print('Processing image set...')
        j=0
        if i > 6000:
            j=1
        outfile = '%s.%d' % (args.outfile, j)
        t = t + len(image_id)
        processed = processed + generate_tsv_mask_only(found_ids, gpu_id, args.prototxt, args.caffemodel, image_id, outfile)
        print(processed)
    print(t)
