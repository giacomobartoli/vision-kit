from distutils.version import StrictVersion

import os
import sys
import tensorflow as tf
import glob
import numpy as np
import tensorflow as tf
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


#sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# For double checking we store predictions on .txt files:
SAVE_LABELS = 'CHECK_LABELS.txt'
SAVE_BBOX = 'CHECK_BBOX.txt'


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = 'frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'core_50_map.pbtxt')

NUM_CLASSES = 50

class_errs = 0
localiz_errs = 0
tot_images = 0


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
print("Loading label maps: "+PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

list_predicted_labels = []
list_predicted_bbox = []
file1 = open(SAVE_LABELS, 'w')
file2 = open(SAVE_BBOX, 'w')

def load_image_into_numpy_array(image_path):
  image = Image.open(image_path)
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def run_inference_for_images(images, graph):
    with graph.as_default():
        with tf.Session() as sess:
            output_dict_array = []

            for image in images:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                image2 = load_image_into_numpy_array(image)
                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image2, 0)})
                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = str(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                output_dict_array.append(output_dict)
                global tot_images
                tot_images = tot_images - 1
                print('Remaining images: ' + str(tot_images))

    return output_dict_array


# LOADING FROZEN GRAPH
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  print("Loading graph: "+PATH_TO_FROZEN_GRAPH)
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# LOADING TEST SET, SORTED ALPHABETICALLY
images = sorted(glob.glob("images/test/*.jpg"))
tot_images=len(images)
print("Test set dimension: "+str(tot_images)+" images")

# ARRAY WITH DETECTION RESULTS
output_dict_array = run_inference_for_images(images, detection_graph)

# CHECKING RESULTS
for idx in range(len(output_dict_array)):
    output_dict = output_dict_array[idx]
    image_np=load_image_into_numpy_array(images[idx])
    image_np_expanded = np.expand_dims(image_np, axis=0)
    #list_predicted_labels.append(vis_util.predicted_label)
    list_predicted_labels.append(vis_util.predicted_label)
    #conversion to original coordinates
    ymin = int(vis_util.predicted_bbox[0] * 350)
    xmin = int(vis_util.predicted_bbox[1] * 350)
    ymax = int(vis_util.predicted_bbox[2] * 350)
    xmax = int(vis_util.predicted_bbox[3] * 350)

    list_predicted_bbox.append(str(ymin) + ',' + str(xmin) + ',' + str(ymax) + ',' + str(xmax))
    file1.write(vis_util.predicted_label+'\n')
    file2.write(str(ymin) + ',' + str(xmin) + ',' + str(ymax) + ',' + str(xmax) + '\n')
    print("Predicted class for " + images[idx] + " --> " + vis_util.predicted_label)
    print("Predicted bbox for " + images[idx] + ": " + str(ymin) + ',' + str(xmin) + ',' + str(ymax) + ',' + str(xmax))


#loading GT labels/bbox from .txt
ground_truth_list_labels = open('images/ground_truth_labels2.txt').read().splitlines()
ground_truth_list_bbox = open('images/ground_truth_bbox2.txt').read().splitlines()

#saving txt files
file1.close()
file2.close()

# check two lists are of the same length:
if len(ground_truth_list_bbox) != len(list_predicted_bbox):
    raise ValueError('LISTE BBOX DI LUNGHEZZA DIFFERENTE!')

if len(ground_truth_list_labels) != len(list_predicted_labels):
    raise ValueError('LISTE LABELS DI LUNGHEZZA DIFFERENTE!')

# calculate class_errors
for f, b in zip(ground_truth_list_labels, list_predicted_labels):
    if f != b:
        class_errs = class_errs+1

# calculate localization errors:
for f, b in zip(ground_truth_list_bbox, list_predicted_bbox):
    f = f.split(",")
    b = b.split(",")
    if bb_intersection_over_union(list(map(int, f)),list(map(int, b))) > 0.5:
        localiz_errs = localiz_errs+1


print('CLASSIFICATION: '+str(class_errs))
print('LOCALIZATION: '+str(localiz_errs))