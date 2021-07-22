# utils.py

This file contain some utilities function for model training.



**import**

```python
 import cv2
 import os
 import shutil
 import sys
 import numpy as np
 import tensorflow as tf
 import colorsys
 from operator import itemgetter
 
 import os
 from model import YOLOv1
```



**define function**

- [dir_setting](#dir_setting)
- [set_checkpoint_manager](#set_checkpoint_manager)

- [draw_bounding_box_and_label_info](#draw_bounding_box_and_label_info)

- [find_max_confidence_bounding_box](#find_max_confidence_bounding_box)
- [yolo_format_to_bounding_box_dict](#yolo_format_to_bounding_box_dict)
- [iou](#iou)
- [generate_color](#generate_color)
- [Full code](#"Full code")

## dir_setting

training이 unintentionally하게 중단되었거나, data의 update로 인해 model을 새롭게 training해야 하는 경우를 위해 directory를 생성 또는 관리하는 function이다.



**input argument description**

- `dir_name` : model 또는 result parameter를 저장하기 위한 direcotry의 name

- `CONTINUE_LEARNING` : training을 이어서 할지, 새롭게 시작할지를 결정하는 variable이다.

  > if `CONTINUE_LEARNING` is `True` but nothing in model directory, `CONTINUE_LEARNING` is `False`

```python
 def dir_setting(dir_name, 
                 CONTINUE_LEARNING, 
                 checkpoint_path, 
                 tensorboard_log_path):
 
     model_path = os.path.join(os.getcwd() , dir_name)
     checkpoint_path = os.path.join(model_path, checkpoint_path)
     tensorboard_log_path = os.path.join(model_path, tensorboard_log_path)
 
     if CONTINUE_LEARNING == True and not os.path.isdir(model_path):
         CONTINUE_LEARNING = False
         print("CONTINUE_LEARNING flag has been converted to FALSE") 
 
     if CONTINUE_LEARNING == False and os.path.isdir(model_path):
         while True:
             print("\n Are you sure remove all directory and file for new training start?  [Y/N] \n")
             answer = str(input())
             if answer == 'Y' or answer == 'y':
                 shutil.rmtree(model_path)
                 break
             elif answer == 'N' or answer == 'n':
                 print("Check 'CONTINUE_LEARNING' in main.py")
                 sys.exit()
             else :
                 print("wrong answer. \n Please enter any key ")
                 tmp = str(input())
                 os.system('clear')  # cls in window 
 
	# set tensorboard log
	train_summary_writer = tf.summary.create_file_writer(tensorboard_log_path +  '/train')
	validation_summary_writer = tf.summary.create_file_writer(tensorboard_log_path +  '/validation')  

	# pass if the path exist. or not, create directory on path
	if not os.path.isdir(model_path):
		os.makedirs(model_path, exist_ok=True)
		os.mkdir(checkpoint_path)

 
     return checkpoint_path, train_summary_writer, validation_summary_writer
```



**detail**

- line 14 : delete all file on `model_path`, if left some file or information when the training start at first

- result; directory structure

  ```
   dir_name    |-- saved_model
               |-- tensorboard_log |-- train
                                   |-- validation
  ```

  

## set_checkpoint_manager

function for managing checkpoint



**input argument description**

- `input_height`, `input_width`,  `cell_size`, `boxes_per_cell`, `num_classes` : parameter for call model instance at line 9

- `checkpoint_path` :  specific directory path for save the checkpoint of model trining

  

```python
 def set_checkpoint_manager(input_height,
                             input_width,
                             cell_size,
                             boxes_per_cell,
                             num_classes,
                             checkpoint_path):
 
     # create YOLO model
     YOLOv1_model = YOLOv1(input_height, input_width, cell_size, boxes_per_cell, num_classes)
 
     # set checkpoint manager
     ckpt = tf.train.Checkpoint(step=tf.Variable(0), model=YOLOv1_model)
     ckpt_manager = tf.train.CheckpointManager(ckpt,
                                             directory=checkpoint_path,
                                             max_to_keep=None)
     latest_ckpt = tf.train.latest_checkpoint(checkpoint_path)
 
     # restore latest checkpoint
     if latest_ckpt:
         ckpt.restore(latest_ckpt)
         print('global_step : {}, checkpoint is restored!'.format(int(ckpt.step)))
     return ckpt, ckpt_manager, YOLOv1_model
```



**detail**

- line 12 : initial entry step 설정은 0 

  > 만약 `CONTINUE_LEARNING` = True 이고 last point가 존재한다면 line16에서 last point를 가져온다.

- line 16 

  `latest_checkpoint` : 마지막 checkpoint에서 저장된 file의 path를 return 

- line 20 : 마지막 checkpoint의 값들을 ckpt에 저장



## save_checkpoint

save model parameters when step(`ckpt.step`) reach specific check point(`FLAGS.save_checkpoint_steps`)



```python
 def save_checkpoint(ckpt, ckpt_manager, save_checkpoint_steps):
     # save checkpoint
     if ckpt.step % save_checkpoint_steps == 0:
         ckpt_manager.save(checkpoint_number=ckpt.step)  # CheckpointManager의 parameter를 저장
         print('global_step : {}, checkpoint is saved!'.format(int(ckpt.step)))
```



## draw_bounding_box_and_label_info

braw bounding box and show text about label information



**input argument description**

- `frame` : ground truth image
- `x_min`,`y_min`, `x_max`, `y_max` : ground truth bounding box information parameter
- `label` : ground truth class name
- `confidence` : ground truth confidence
- `color` : instance of the function defined`generate_color`

```python
 def draw_bounding_box_and_label_info(frame, x_min, y_min, x_max, y_max, label, confidence, color):
     # draw rectangle
     cv2.rectangle(
         frame,
         (x_min, y_min),
         (x_max, y_max),
         color, 3)
 
     # draw label information
     text = label + ' ' + str('%.3f' % confidence)
     bottomLeftCornerOfText = (x_min, y_min)
     font = cv2.FONT_HERSHEY_SIMPLEX
     fontScale = 0.8
     fontColor = color
     lineType = 2
 
     cv2.putText(frame, text,
                 bottomLeftCornerOfText,
                 font,
                 fontScale,
                 fontColor,
                 lineType)
```



**detail**

- line 3

  `cv2.rectangle` : image위에 사각형 그리기

  ```
   cv2.rectangle(img, pt1, pt2, color, thickness = None, lineType = None, shift =None )
  ```

  `pt1, pt2` : 사각형의 두 꼭지점(좌측 상단, 우측 하단) 좌표

- line 24 

  `cv2.putText` : image에 text 삽입 method

  ```
   cv2.putText(img, text, org, fontFace, fontScale, color, thickness = , lineType =, bottomLeftOrigin = )
  ```

  `text` : 출력할 문자열

  `org` : text의 위치 좌측 하단 좌표

  `fontFace` : font 종류  

  `fontScale` : font size

  `bottomLeftOrigin` : Ture or False.

  `lineType` : 선형 타입.  cv2.LINE_AA 또는 2



## find_max_confidence_bounding_box

find one max confidence bounding box of the entire bounding boxes

>  Implement abbreviated version non-*maximum* suppression



sorted : key를 기준으로 가장 높은 값을 가진 것을 차례로 정렬 후 반환

```python
 def find_max_confidence_bounding_box(bounding_box_info_list):
     bounding_box_info_list_sorted = sorted(bounding_box_info_list,
                                             key=itemgetter('confidence'),
                                             reverse=True)
     # 전체 bounding box 내림차순 sorting
     max_confidence_bounding_box = bounding_box_info_list_sorted[0]
 
     # confidence가 가장 높은 bounding box return
     return max_confidence_bounding_box
```





## yolo_format_to_bounding_box_dict

create dictionary of bounding box information 



**input argument description**

- `xcenter` :  x center coordinates of bounding box
- `ycenter` : y center coordinates of bounding box
- `box_w` :  width of bounding box
- `box_h` : height of bounding box
- `class_prediction` : class
- `confidence` : normalized confidence 

```python
 def yolo_format_to_bounding_box_dict(xcenter, ycenter, box_w, box_h, class_name, confidence):
     # the zero coordinate of image located
     bounding_box_info = dict()
     bounding_box_info['left'] = int(xcenter - (box_w / 2))
     bounding_box_info['top'] = int(ycenter + (box_h / 2))
     bounding_box_info['right'] = int(xcenter + (box_w / 2))
     bounding_box_info['bottom'] = int(ycenter - (box_h / 2))
     bounding_box_info['class_name'] = class_name
     bounding_box_info['confidence'] = confidence
 
     return bounding_box_info
```





## iou

calculate **IOU**(Intersection over Union)



**Intersection over Union**

![img](https://t1.daumcdn.net/cfile/tistory/993477505D14A25016)

IoU = 교집합 영역 넓이 / 합집합 영역 넓이

- Reference : https://github.com/nilboy/tensorflow-yolo/blob/python2.7/yolo/net/yolo_tiny_net.py#L105



**input argument description**

- `yolo_pred_boxes` : predicted bounding boxes information

  > 4-D tensor [cell_size, cell_size, boxes_per_cell, 4]

- `ground_truth_boxes` : label bounding boxes information

  > 1-D tensor [x_center, y_center, w, h]



```python
def iou(yolo_pred_boxes, ground_truth_boxes):
 
     boxes1 = yolo_pred_boxes
     boxes2 = ground_truth_boxes
 
     boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                        boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
     boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])
     boxes2 = tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
                        boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])
     boxes2 = tf.cast(boxes2, tf.float32)
 
     # calculate the left up point
     lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2]) # 두 Bbox 중 x, y의 최대값
     rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:]) # 두 Bbox 중 w, h의 최소값
   
     # intersection
     intersection = rd - lu
 
     inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]
 
     mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)
 
     inter_square = mask * inter_square # 교집합
 
     # calculate the boxs1 square and boxs2 square
     square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
     square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])
 
     iou = inter_square / (square1 + square2 - inter_square + 1e-6)
     # '1e-6' : for the denominator to not zero
 
     # return value range is 0 ~ 1
     return iou
```





## generate_color



Generate each colors about each class for drawing bounding boxes 

Each color is determined randomly

- Reference : https://github.com/qqwweee/keras-yolo3/blob/e6598d13c703029b2686bc2eb8d5c09badf42992/yolo.py#L82



```python
 def generate_color(num_classes):
     hsv_tuples = [(x / num_classes, 1., 1.)
                    for x in range(num_classes)]
     colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
     colors = list(
         map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
 
     np.random.seed(10101)       # Fixed seed for consistent colors across runs.
     np.random.shuffle(colors)   # Shuffle colors to decorrelate adjacent classes.
     np.random.seed(None)        # Reset seed to default.
 
     return colors
```





## Full code

```python
import cv2
import os
import shutil
import sys
import numpy as np
import tensorflow as tf
import colorsys
from operator import itemgetter
 
import os
from model import YOLOv1
 
 
def dir_setting(dir_name, 
                 CONTINUE_LEARNING, 
                 checkpoint_path, 
                 tensorboard_log_path):
 
     model_path = os.path.join(os.getcwd() , dir_name)
     checkpoint_path = os.path.join(model_path, checkpoint_path)
     tensorboard_log_path = os.path.join(model_path, tensorboard_log_path)
 
     if CONTINUE_LEARNING == True and not os.path.isdir(model_path):
         CONTINUE_LEARNING = False
         print("CONTINUE_LEARNING flag has been converted to FALSE") 
 
     if CONTINUE_LEARNING == False and os.path.isdir(model_path):
         while True:
             print("\n Are you sure remove all directory and file for new training start?  [Y/N] \n")
             answer = str(input())
             if answer == 'Y' or answer == 'y':
                 shutil.rmtree(model_path)
                 break
             elif answer == 'N' or answer == 'n':
                 print("Check 'CONTINUE_LEARNING' in main.py")
                 sys.exit()
             else :
                 print("wrong answer. \n Please enter any key ")
                 tmp = str(input())
                 os.system('clear')  # cls in window 
 
	# set tensorboard log
	train_summary_writer = tf.summary.create_file_writer(tensorboard_log_path +  '/train')
	validation_summary_writer = tf.summary.create_file_writer(tensorboard_log_path +  '/validation')  

	# pass if the path exist. or not, create directory on path
	if not os.path.isdir(model_path):
		os.makedirs(model_path, exist_ok=True)
		os.mkdir(checkpoint_path)
 
     return checkpoint_path, train_summary_writer, validation_summary_writer
 
 
 def set_checkpoint_manager(input_height,
                             input_width,
                             cell_size,
                             boxes_per_cell,
                             num_classes,
                             checkpoint_path):
 
     # create YOLO model
     YOLOv1_model = YOLOv1(input_height, input_width, cell_size, boxes_per_cell, num_classes)
 
     # set checkpoint manager
     ckpt = tf.train.Checkpoint(step=tf.Variable(0), model=YOLOv1_model)
     ckpt_manager = tf.train.CheckpointManager(ckpt,
                                             directory=checkpoint_path,
                                             max_to_keep=None)
     latest_ckpt = tf.train.latest_checkpoint(checkpoint_path)
 
     # restore latest checkpoint
     if latest_ckpt:
         ckpt.restore(latest_ckpt)
         print('global_step : {}, checkpoint is restored!'.format(int(ckpt.step)))
     return ckpt, ckpt_manager, YOLOv1_model
 
 
 def save_checkpoint(ckpt, ckpt_manager, save_checkpoint_steps):
     # save checkpoint
     if ckpt.step % save_checkpoint_steps == 0:
         ckpt_manager.save(checkpoint_number=ckpt.step)  # CheckpointManager의 parameter를 저장
         print('global_step : {}, checkpoint is saved!'.format(int(ckpt.step)))
         
         
 def draw_bounding_box_and_label_info(frame, x_min, y_min, x_max, y_max, label, confidence, color):
     # draw rectangle
     cv2.rectangle(
         frame,
         (x_min, y_min),
         (x_max, y_max),
         color, 3)
 
     # draw label information
     text = label + ' ' + str('%.3f' % confidence)
     bottomLeftCornerOfText = (x_min, y_min)
     font = cv2.FONT_HERSHEY_SIMPLEX
     fontScale = 0.8
     fontColor = color
     lineType = 2
 
     cv2.putText(frame, text,
                 bottomLeftCornerOfText,
                 font,
                 fontScale,
                 fontColor,
                 lineType)
     
 
 def find_max_confidence_bounding_box(bounding_box_info_list):
     bounding_box_info_list_sorted = sorted(bounding_box_info_list,
                                             key=itemgetter('confidence'),
                                             reverse=True)
     # 전체 bounding box 내림차순 sorting
     max_confidence_bounding_box = bounding_box_info_list_sorted[0]
 
     # confidence가 가장 높은 bounding box return
     return max_confidence_bounding_box
 
 
 def yolo_format_to_bounding_box_dict(xcenter, ycenter, box_w, box_h, class_name, confidence):
     # the zero coordinate of image located
     bounding_box_info = dict()
     bounding_box_info['left'] = int(xcenter - (box_w / 2))
     bounding_box_info['top'] = int(ycenter + (box_h / 2))
     bounding_box_info['right'] = int(xcenter + (box_w / 2))
     bounding_box_info['bottom'] = int(ycenter - (box_h / 2))
     bounding_box_info['class_name'] = class_name
     bounding_box_info['confidence'] = confidence
 
     return bounding_box_info
 
 
 
 def iou(yolo_pred_boxes, ground_truth_boxes):
 
     boxes1 = yolo_pred_boxes
     boxes2 = ground_truth_boxes
 
     boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                        boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
     boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])
     boxes2 = tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
                        boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])
     boxes2 = tf.cast(boxes2, tf.float32)
 
     # calculate the left up point
     lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
     rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])
   
     # intersection
     intersection = rd - lu
 
     inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]
 
     mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)
 
     inter_square = mask * inter_square
 
     # calculate the boxs1 square and boxs2 square
     square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
     square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])
 
     iou = inter_square / (square1 + square2 - inter_square + 1e-6)
     # '1e-6' : for the denominator to not zero
 
     # return value range is 0 ~ 1
     return iou
 
 
 def generate_color(num_classes):
     hsv_tuples = [(x / num_classes, 1., 1.)
                    for x in range(num_classes)]
     colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
     colors = list(
         map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
 
     np.random.seed(10101)       # Fixed seed for consistent colors across runs.
     np.random.shuffle(colors)   # Shuffle colors to decorrelate adjacent classes.
     np.random.seed(None)        # Reset seed to default.
 
     return colors
```

