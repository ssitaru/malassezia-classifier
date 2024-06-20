"""
    Python script for splitting image data to train and test data per class
    80/20 split

    Sebastian Sitaru
"""
import os
import argparse
import random
from shutil import copy

ap = argparse.ArgumentParser(description='Split images.')
ap.add_argument('path', metavar='P', type=str, help='path to images/{class1, ...}')
ap.add_argument("-o", "--output-path", help="Path to create (default: data.split)", default="data.split")
args = ap.parse_args()
img_path = args.path
out_path = args.output_path

IMGS = []
for cls in os.listdir(img_path):
    IMGS = []
    print('class', cls)
    
    for f in os.listdir(os.path.join(img_path, cls)):
        IMGS.append(os.path.join(img_path, cls, f))

    # Split images in train and test data
    random.shuffle(IMGS)
    split_1 = int(0.8*len(IMGS))
    train_imgs = IMGS[:split_1]
    test_imgs = IMGS[split_1:]
    print('total:', len(IMGS))
    print('train:', len(train_imgs))
    print('test:', len(test_imgs))
    os.makedirs(out_path + '/train/'+cls, exist_ok=True)
    os.makedirs(out_path + '/test/'+cls, exist_ok=True)
    for img in train_imgs:
        fr = img
        to = out_path + '/train/'+cls+'/'+os.path.basename(img)
        copy(img, to)
        print('copied '+fr+' to '+to)
        #pass
    for img in test_imgs:
        fr = img
        to = out_path + '/test/'+cls+'/'+os.path.basename(img)
        copy(fr, to)
        print('copied '+fr+' to '+to)
        #pass 



