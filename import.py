#!/usr/bin/env python
import os
import sys
import subprocess
import simplejson as json
import numpy
import cv2
from picpac import Writer
from ljpeg import ljpeg

D_X = [0, 1, 1, 1, 0, -1, -1, -1]
D_Y = [-1, -1, 0, 1, 1, 1, 0, -1]

MAX_IMPORT = -1 #10
MAX_SIZE = -1 #1000
imported = 0

CODEC = '.exr'
#CODEC = '.jpg'

# sample boundary
# 200 1888 2 2 2 2 2 2 2 2 2 #
# convert to polygon annotation
def make_poly (line, H, W):
    E = []
    for x in line.strip().split(' '):
        try:
            E.append(int(x))
        except:
            pass
        pass
    x, y = E[:2]
    cc = []
    for d in E[2:-1]:
        cc.append({'x':1.0 * float(x) / W, 'y':1.0 * y / H})
        if d == '#':
            break
        x += D_X[d]
        y += D_Y[d]
        pass
    return {'type':'polygon', 'geometry':{'points': cc}}

#ICS sample:
#ics_version 1.0
#filename B-3159-1
#DATE_OF_STUDY 29 5 1998
#PATIENT_AGE 58
#FILM
#FILM_TYPE REGULAR
#DENSITY 2
#DATE_DIGITIZED 14 7 1998
#DIGITIZER LUMISYS LASER
#SEQUENCE
#LEFT_CC LINES 4544 PIXELS_PER_LINE 3120 BITS_PER_PIXEL 12 RESOLUTION 50 OVERLAY
#LEFT_MLO LINES 4536 PIXELS_PER_LINE 3112 BITS_PER_PIXEL 12 RESOLUTION 50 OVERLAY
#RIGHT_CC LINES 4616 PIXELS_PER_LINE 3088 BITS_PER_PIXEL 12 RESOLUTION 50 NON_OVERLAY
#RIGHT_MLO LINES 4616 PIXELS_PER_LINE 3088 BITS_PER_PIXEL 12 RESOLUTION 50 NON_OVERLAY

def load_sample (db, base, H, W):
    i_path = base + '.LJPEG'
    o_path = base + '.OVERLAY'
    if not (os.path.exists(o_path) and os.path.exists(i_path)):
        return
    print 'Loading %s with (%dx%d)' % (base, H, W)
    image = ljpeg.read(i_path).astype('float').reshape(H, W)
    H, W = image.shape
    # !!! Need better color scaling
    M = numpy.max(image)
    image *= 255 / M
    if MAX_SIZE > 0:
        m = max(image.shape)
        if m > MAX_SIZE:
            r = 1.0 * MAX_SIZE / m
            image = cv2.resize(image, None, None, r, r)
    shapes = []
    with open(o_path, 'r') as f:
        try:
            while True:
                x = f.next()
                x = x.strip()
                if x == "BOUNDARY":
                    l = f.next()
                    l = l.strip()
                    shapes.append(make_poly(l, H, W))
                    pass
                pass
        except StopIteration:
            pass
        pass
    print "SIZE: ", image.shape
    rv, image = cv2.imencode(CODEC, image)
    if not rv:
        return
    image = image.tostring()
    anno = json.dumps({'shapes': shapes})
    db.append(image, anno)
    global imported
    imported += 1
    pass

def load_ics (db, ics):
    # sample ics path
    # ...e3159/B-3159-1.ics
    dir = os.path.dirname(ics)
    # base name of ics, ==> B_3159_1
    bn = os.path.splitext(os.path.basename(ics))[0]
    bn = bn.replace('-', '_')

    with open(ics, 'r') as f:
        while f.next().strip() != 'SEQUENCE':
            pass
        for l in f:
            l = l.strip()
            fs = l.split(' ')
            if fs[-1] != 'OVERLAY':
                # in the future we might want to keep those exmpty samples
                continue
            base = os.path.join(dir, bn + '.' + fs[0])
            H = int(fs[2])
            W = int(fs[4])
            load_sample(db, base, H, W)
            pass
        pass
    pass

try:
    os.remove('db')
except:
    pass
db = Writer('db')

DIR = '.'
if len(sys.argv) > 1:
    DIR = sys.argv[1]

lines = subprocess.check_output("find %s -name '*.ics'" % DIR, shell=True)
for line in lines.split('\n'):
    if MAX_IMPORT > 0 and imported >= MAX_IMPORT:
        break
    line = line.strip()
    if len(line) == 0:
        continue
    load_ics(db, line)
    pass

