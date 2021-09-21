from os import listdir
from os.path import isfile, join
import argparse
#import cv2
import numpy as np
import sys
import os
import shutil
from data_aug.data_aug import *
from data_aug.bbox_util import *
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# stats: {'Person': 14979, 'Cyclist': 2382, 'Motorcyclist': 3826, 'Bus': 1770, 'Car': 23157, 'Truck': 1071}

def check_balance(label_stats):
    min_key = min(label_stats, key=label_stats.get)
    max_key = max(label_stats, key=label_stats.get)
    print("max_key: {}, has {}, min_key: {}, has {}".format(max_key, min_key, label_stats[max_key], label_stats[min_key]))
    if (label_stats[max_key] - label_stats[min_key]) > 0.2 * label_stats[max_key]:
        return False
    else:
        return True

def augment_labels(lines):
    transforms = Sequence([RandomHorizontalFlip(1), RandomScale(0.2, diff = True), RandomRotate(10)])
    for line in lines:
        bboxes = []
        #line = line.replace('JPEGImages','labels')
        img = cv2.imread(line)[:,:,::-1] #OpenCV uses BGR channels
        height = img.shape[0]
        width = img.shape[1]
        original_image_path = line
        line = line.replace('.jpg','.txt')
        line = line.replace('.png','.txt')
        image_path_parts = original_image_path.split('/')
        label_path_parts = line.split('/')
        parent_path = "/".join(line.split('/')[:-1])
        path_len = len(image_path_parts)
        image_path_parts[path_len-2] = "images_augmented"
        image_path = "/".join(image_path_parts)
        label_path_parts[path_len-2] = "images_augmented"
        label_path_parts[-1] = label_path_parts[-1].replace('.txt', '_augmented.txt')
        label_path = "/".join(label_path_parts)

        print(line)
        f2 = open(line)
        for line in f2.readlines():
            items = line.rstrip('\n').split(' ')
            #print("items: {}".format(items))
            class_id = int(items[0])
            label_stats[class_names[class_id]] += 1
            center_x_ratio = float(items[1])
            center_y_ratio = float(items[2])
            w_ratio = float(items[3])
            h_ratio = float(items[4])
            bboxes.append([width * max(0, center_x_ratio - 0.5*w_ratio),
                           height * max(0, center_y_ratio - 0.5*h_ratio),
                           width * min(width, center_x_ratio + 0.5*w_ratio),
                           height * min(height, center_y_ratio + 0.5*h_ratio),
                           class_id])
        if len(bboxes) > 0:
            bboxes = np.array(bboxes)
            img, bboxes = transforms(img, bboxes)
            Path(parent_path).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            with open(label_path, 'w') as aug_label_file:
                for box in bboxes:
                    center_x_ratio = float(box[0] + box[2]) / (2*width)
                    center_y_ratio = float(box[1] + box[3]) / (2*height)
                    w_ratio = float(box[2] - box[0])/width
                    h_ratio = float(box[3] - box[1])/height
                    aug_label_file.write(str(box[4]) + " " + str(center_x_ratio)
                        + " " + str(center_y_ratio) + " " + str(w_ratio)
                        + " " + str(h_ratio) + '\n')
            plt.imshow(draw_rect(img, bboxes))
            #plt.show()
    return

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-file_list', default = 'train.txt',
                        help='path to filelist\n' )
    parser.add_argument('-class_file', default = 'bus.names', type = str,
                        help='The class names\n' )
    args = parser.parse_args()

    f = open(args.class_file)
    class_names = [line.rstrip('\n') for line in f.readlines()]
    class_name_id_dict = {}
    id = 0
    for name in class_names:
        class_name_id_dict[name] = id
        id += 1

    f = open(args.file_list)
    lines = [line.rstrip('\n') for line in f.readlines()]

    label_stats = {}
    for i in range(len(class_names)):
        label_stats[class_names[i]] = 0
    for line in lines:

        line = line.replace('.jpg','.txt')
        line = line.replace('.png','.txt')

        print(line)
        f2 = open(line)
        for line in f2.readlines():
            items = line.rstrip('\n').split(' ')
            #print("items: {}".format(items))
            class_id = int(items[0])
            label_stats[class_names[class_id]] += 1


    print("stats: {}".format(label_stats))
    plt.bar(label_stats.keys(), label_stats.values(), 1, color='g')
    plt.show()
    if not check_balance(label_stats):
        print("Unbalanced dataset!")
        augment_labels(lines, class_name_id_dict)


if __name__=="__main__":
    main(sys.argv)
