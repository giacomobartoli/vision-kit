## Snippet
Few lines of useful code:


# Tensorflow Object Detection API

training:
`python object_detection/eval.py 
 --logtostderr 
 --pipeline_config_path=object_”detection/samples/configs/faster_rcnn_resnet101_voc07.config” 
 --checkpoint_dir=“VOCdevkit/VOC2012" 
  --eval_dir=“VOCdevkit/eval”`
  
eval:
`python object_detection/train.py     
--logtostderr     
--pipeline_config_path="object_detection/samples/configs/faster_rcnn_resnet101_voc07.config"    
--train_dir="VOCdevkit/VOC2012"`

inspect graph:

`pip install tensorboard`
`ensorboard --logdir=/tmp/tensorboard`
`python tensorflow/python/tools/import_pb_to_tensorboard.py --model_dir resnetv1_50.pb --log_dir /tmp/tensorboard`

# Vision Kit

Compile
`./bonnet_model_compiler.par \
  --frozen_graph_path=frozen_inference_graph.pb \
  --output_graph_path=frozen_inference_graph.binaryproto \
  --input_tensor_name=input \
  --output_tensor_names=final_result \
  --input_tensor_size=256`
  
Execute

`~/AIY-projects-python/src/examples/vision/mobilenet_based_classifier.py \
  --model_path ~/models/mobilenet_v2_192res_1.0_inat_plant.binaryproto \
  --label_path ~/models/mobilenet_v2_192res_1.0_inat_plant_labels.txt \
  --input_height 192 \
  --input_width 192 \
  --input_layer map/TensorArrayStack/TensorArrayGatherV3 \
  --output_layer prediction \
  --preview`