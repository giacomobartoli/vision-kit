import csv
import re
import os
from fnmatch import fnmatch

# header for the .csv file
column_name = ['Filename', 'width', 'height', 'session', 'num_obj','frame', 'xmin', 'ymin', 'xmax', 'ymax','label' ]

# CORe50 images width and height
C50_W = 350
C50_H = 350

# CORe50 root
root = 'core50_350x350/'
pattern = "*.png"

# Bounding boxes root
bbox = 'bbox/'
pattern_bbox = "*.txt"

# This is an empty list that will be filled with all the data: filename, width, height, session..etc
filenames = []

# some regex used for finding session, obj, frame
re_find_session = '(?<=.{2}).\d'
re_find_object = '(?<=.{5}).\d'
re_find_frame = '(?<=.{8})..\d'


def find_obj(s, regex):
    obj = re.search(regex, s)
    return obj.group()


def find_bbox(session, obj, frame):
    bb_path = 'bbox/'+session+'/'+'CropC_'+obj+'.txt'
    f = open(bb_path, 'r').readlines()
    for line in f:
        regex_frame = '[Color'+frame+': ][0-9]*'
        bb_pattern = re.compile(regex_frame)
        c = bb_pattern.match(line).string.strip()
        return c[10:]


# c[0] = xmin, c[1] = ymin, c[2] = xmax, c[3] = ymax
def add_bbox_to_list(bbox, list):
    c = bbox.split()
    list.append(c[0])
    list.append(c[1])
    list.append(c[2])
    list.append(c[3])
    return list

# given an object, it returns the label
def add_class_to_list(object):
    index = int(object[1:])
    f = open('core50_class_names.txt', 'r').readlines()
    return f[index-1].strip()

# scanning the file system, creating a list with all the data
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            listToAppend = []
            listToAppend.append(name)
            listToAppend.append(C50_H)
            listToAppend.append(C50_W)
            session = 's' + find_obj(name, re_find_session).strip('0')
            object = 'o' + find_obj(name, re_find_object).strip('0')
            frame = find_obj(name, re_find_frame)
            listToAppend.append(session)
            listToAppend.append(object)
            listToAppend.append(frame)
            bounding_box = find_bbox(session, object, frame)
            add_bbox_to_list(bounding_box, listToAppend)
            listToAppend.append(add_class_to_list(object))
            # print(name)
            filenames.append(listToAppend)

print(filenames)

# writing data to the .csv file
with open('core50.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(column_name),
    for i in filenames:
        filewriter.writerow(i)

print ('Done! Your .csv file is ready!')

