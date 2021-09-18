from os import listdir
from os.path import isfile, join
import argparse
#import cv2
import numpy as np
import sys
import os
import shutil
import matplotlib.pyplot as plt



def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-file_list', default = 'train.txt',
                        help='path to filelist\n' )
    parser.add_argument('-class_file', default = 'bus.names', type = str,
                        help='The class names\n' )
    args = parser.parse_args()

    f = open(args.class_file)
    class_names = [line.rstrip('\n') for line in f.readlines()]

    f = open(args.file_list)
    lines = [line.rstrip('\n') for line in f.readlines()]

    label_stats = {}
    for i in range(len(class_names)):
        label_stats[class_names[i]] = 0
    for line in lines:
        #line = line.replace('JPEGImages','labels')
        line = line.replace('.jpg','.txt')
        line = line.replace('.png','.txt')
        print(line)
        f2 = open(line)
        for line in f2.readlines():
            class_id = int(line.rstrip('\n').split(' ')[0])
            label_stats[class_names[class_id]] += 1
    print("stats: {}".format(label_stats))
    plt.bar(label_stats.keys(), label_stats.values(), 1, color='g')
    plt.show()

if __name__=="__main__":
    main(sys.argv)
