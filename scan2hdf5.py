#!/usr/bin/env python
import os
import logging
import numpy as np
import cv2
import h5py

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

def save_hd5py (out_path, data, folds = 0):
    images = np.concatenate([a[0] for a in data], axis = 0)
    labels = np.concatenate([a[1] for a in data], axis = 0)
    f = h5py.File(out_path, mode='w')
    ds = f.create_dataset('images', images.shape, dtype=str(images.dtype))
    ds[...] = images
    ds = f.create_dataset('labels', labels.shape, dtype=str(labels.dtype))
    ds[...] = labels
    #assert(folds > 1)
    #if folds > 1:
    fold = len(images) // folds
    idx = {'fold-{}'.format(i): (i*fold, (i+1)*fold) for i in range(folds)}
    print idx
    split_dict = {k: {'images': v, 'labels':v} for k, v in idx.iteritems()}
    from fuel.datasets.hdf5 import H5PYDataset
    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    f.flush()
    f.close()
    pass

def scan2hdf5 (out_path, image_dir, label_dir, folds=0, resize=255, gray=False, useall=False):
    images = find_images(image_dir)
    labels = find_images(label_dir)
    all = []
    chs = 1 if gray else 3
    for key, ipath in images.iteritems():
        image = cv2.imread(ipath, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        image = cv2.resize(image, (resize, resize)).reshape(1, chs, resize, resize)
        lpath = labels.get(key, None)
        if lpath:
            label = cv2.imread(lpath, cv2.IMREAD_GRAYSCALE)
            label = cv2.resize(label, (resize, resize)).reshape(1, 1, resize, resize)
        elif useall:    # no label, set all negative
            label = np.zeros((1, 1, resize, resize), dtype=np.uint8)
        else:
            logging.warning('no label found for {}'.format(ipath))
            continue
        all.append((image, label))
        pass
    logging.info("found {} images".format(len(all)))
    save_hd5py(out_path, all, folds)
    pass

if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='import training images & labels')
    #parser.add_argument('--root', default='hdfs://washtenaw:19000/user/hive/warehouse/wdong_tri.db')
    parser.add_argument('output', nargs=1)
    parser.add_argument('image', nargs=1)
    parser.add_argument('label', nargs=1)
    parser.add_argument('--folds', default=3, type=int)
    parser.add_argument('--resize', default=256, type=int)
    parser.add_argument('--gray', default=False, action='store_true')
    parser.add_argument('--all', default=False, action='store_true')
    args = parser.parse_args()
    scan2hdf5(args.output[0], args.image[0], args.label[0], args.folds, args.resize, args.gray, args.all)
    pass

