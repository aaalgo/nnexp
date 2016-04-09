#!/usr/bin/env python
import os
import logging
import numpy as np
import cv2
import picpac

# recognized image types
IMAGE_EXTS = set(['.jpg', '.png', '.pgm', '.pnm'])

# find all images
# return {basename: path}
def find_images (dir):
    all = {}
    for root, dirs, files in os.walk(dir):
        for f in files:
            base, ext = os.path.splitext(f)
            if not ext.lower() in IMAGE_EXTS:
                continue
            all[base] = os.path.join(root, f)
            pass
        pass
    return all

def encode (image):
    OK, buf = cv2.imencode('.png', image)
    assert OK
    return buf.tostring()

def scan2hdf5 (out_path, image_dir, label_dir, resize=255, gray=False, useall=False):
    writer = picpac.Writer(out_path)
    images = find_images(image_dir)
    labels = find_images(label_dir)
    chs = 1 if gray else 3
    cnt = 0
    for key, ipath in images.iteritems():
        image = cv2.imread(ipath, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        image = cv2.resize(image, (resize, resize))
        lpath = labels.get(key, None)
        if lpath:
            label = cv2.imread(lpath, cv2.IMREAD_GRAYSCALE)
            #_,label = cv2.threshold(label, 1,255,cv2.THRESH_BINARY_INV)
            contour_img = lpath
            if 'c_' in lpath:
                _,label = cv2.threshold(label, 1,255,cv2.THRESH_BINARY_INV)
            else:
                _,label = cv2.threshold(label, 127,255,cv2.THRESH_BINARY_INV)
            label = cv2.resize(label, (resize, resize))
        elif useall:    # no label, set all negative
            label = np.zeros((resize, resize), dtype=np.uint8)
        else:
            logging.warning('no label found for {}'.format(ipath))
            continue
        writer.append(encode(image), encode(label))
        cnt += 1
        pass
    logging.info("found {} images".format(cnt))
    pass

if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='import training images & labels')
    #parser.add_argument('--root', default='hdfs://washtenaw:19000/user/hive/warehouse/wdong_tri.db')
    parser.add_argument('output', nargs=1)
    parser.add_argument('image', nargs=1)
    parser.add_argument('label', nargs=1)
    parser.add_argument('--resize', default=256, type=int)
    parser.add_argument('--gray', default=False, action='store_true')
    parser.add_argument('--all', default=False, action='store_true')
    args = parser.parse_args()
    scan2hdf5(args.output[0], args.image[0], args.label[0], args.resize, args.gray, args.all)
    pass

