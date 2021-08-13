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



**function list**

- [dir_setting](#dir_setting)
- [set_checkpoint_manager](#set_checkpoint_manager)
- [draw_bounding_box_and_label_info](#draw_bounding_box_and_label_info)
- [find_max_confidence_bounding_box](#find_max_confidence_bounding_box)
- [yolo_format_to_bounding_box_dict](#yolo_format_to_bounding_box_dict)
- [iou](#iou)
- [generate_color](#generate_color)
- [remove_irrelevant_label](#remove_irrelevant_label)
- [x_y_center_sort](#x_y_center_sort)
- [performance_evaluation](#performance_evaluation)
- [Full code](#"Full code")



---



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

	return (checkpoint_path,
			train_summary_writer,
			validation_summary_writer)
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
		ckpt_manager.save(checkpoint_number=ckpt.step)  
		print('global_step : {}, checkpoint is saved!'.format(int(ckpt.step)))
```



## draw_bounding_box_and_label_info

draw bounding box and show text about label information



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

find one max confidence score bounding box from entire bounding boxes

>  Implement abbreviated version non-*maximum* suppression



```python
def find_confidence_bounding_box(bounding_box_info_list, confidence_threshold):
	bounding_box_info_list_sorted = sorted(bounding_box_info_list,
											key=itemgetter('confidence_score'),
											reverse=True)
	confidence_bounding_box_list = list()
	check = False

	# confidence값이 confidence_threshold 이상인 Bbox는 모두 표현
	for index in range(len(bounding_box_info_list_sorted)):
		if bounding_box_info_list_sorted[index]['confidence_score'] > confidence_threshold:
			confidence_bounding_box_list.append(bounding_box_info_list_sorted[index])
		else : 
			break

	return confidence_bounding_box_list, check
```

> sorted : key를 기준으로 가장 높은 값을 가진 것을 차례로 정렬 후 반환
>
> `confidence_threshold` 값은 train.py에서 정의되었다.



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
def yolo_format_to_bounding_box_dict(xcenter, ycenter, box_w, box_h, class_name, confidence_score, iou):
	# the zero coordinate of image located
	bounding_box_info = dict()
	bounding_box_info['left'] = int(xcenter - (box_w / 2))
	bounding_box_info['top'] = int(ycenter + (box_h / 2))
	bounding_box_info['right'] = int(xcenter + (box_w / 2))
	bounding_box_info['bottom'] = int(ycenter - (box_h / 2))
	bounding_box_info['class_name'] = class_name
	bounding_box_info['confidence_score'] = confidence_score
	bounding_box_info['iou'] = iou

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

	# yolo_pred_boxes의 0:xcenter, 1:ycenter, 2:w, 3:h
	# x-(w/2) == Bbox의 좌측 하단 꼭지점, y+(h/2) == Bbox의 우측 상단 꼭지점
	# boxes1에는 [xmin, ymin, xmax, ymax] 사각형 꼭지점의 x, y좌표 할당
	boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
					   boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
	boxes1 = tf.transpose(boxes1, [1, 2, 3, 0]) 	# shape을 [4 7 7 2] 에서 [7 7 2 4]로 변경

	
	# ground_truth_boxes의 0:xcenter, 1:ycenter, 2:w, 3:h
	boxes2 = tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
					   boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])
	boxes2 = tf.cast(boxes2, tf.float32)


	# calculate the left up point
	lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])  # 교집합 영역의 좌측 하단 꼭지점 좌표
	rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])	# 교집합 영역의 우측 상단 꼭지점 좌표

	# intersectiony
	intersection = rd - lu  

	inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1] # x크기 * y크기 == 넓이

	# 위에서 계산된 intersection 영역 중에서 x, y 모두 0보다 큰 값이 있어야 교집합이 존재.
	mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)

	# 각 cell 마다, 그리고 각 Bbox중 교집합이 존재하는 cell과 box만 True
	inter_square = mask * inter_square

	# calculate the boxs1 square and boxs2 square
	square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1]) # yolo_pred_boxes의 넓이
	square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1]) 									# ground_truth_boxes의 넓이

	iou = inter_square / (square1 + square2 - inter_square + 1e-6)
	# '1e-6' : for the denominator to not zero
	# 교집합 영역이 없으면 iou는 zero
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

	np.random.seed(10101)  		# Fixed seed for consistent colors across runs.
	np.random.shuffle(colors)  	# Shuffle colors to decorrelate adjacent classes.
	np.random.seed(None) 		# Reset seed to default.

	return colors
```



## remove_irrelevant_label

detect할 class에 대한 label만 추려내고, 나머지 label은 0으로 만드는 function



```python
def remove_irrelevant_label(batch_bbox, batch_labels, class_name_dict):
	tmp_labels = np.zeros_like(batch_labels)
	tmp_bbox = np.zeros_like(batch_bbox)

	for i in range(int(tf.shape(batch_labels)[0])): 		# image 1개당
		for j in range(int(tf.shape(batch_labels)[1])):		# object 1개당
			for lable_num in class_name_dict.keys(): 
				if batch_labels[i][j] == lable_num:
					tmp_labels[i][j] = batch_labels[i][j]
					tmp_bbox[i][j] = batch_bbox[i][j]
					continue

	batch_labels = tf.constant(tmp_labels)
	batch_bbox = tf.constant(tmp_bbox)

	return batch_bbox, batch_labels
```



## x_y_center_sort

`performance_evaluation` 에서 **classification_accuracy** 를 계산하는 과정 중 x, y coordinate에 대한 정렬을 수행하는 function



```python
def x_y_center_sort(labels, taget):

	tmp = np.zeros_like(labels)
	if taget == "x":
		label = list(np.array(labels[:, 0]))
	elif taget == "y":
		label = list(np.array(labels[:, 1]))

	origin_label = label.copy()
	label.sort()
	for i_index, i_value in enumerate(label):
		for j_index, j_value in enumerate(origin_label):
			if i_value == j_value:
				tmp[i_index] = labels[j_index]
				continue
	labels = tmp
	
	return labels
```



## performance_evaluation

validation, test result에 대한 performance evaluation를 수행하는 function



```python
def performance_evaluation(confidence_bounding_box_list,
						   object_num,
						   labels,
						   class_name_to_label_dict,
						   validation_image_index,
						   num_classes):

	x_center_sort_labels = None
	y_center_sort_labels = None
	x_center_sort_pred = None
	y_center_sort_pred = None

	pred_list = np.zeros(shape =(object_num, 3))

	correct_answers_class_num = 0.0  # classification accuracy 계산을 위한 값
	success_detection_num = 0.0 # perfect detection accuracy 계산을 위한 값

	# label object 중 detection한 object의 비율
	detection_rate = len(confidence_bounding_box_list)/object_num  
	#print(f"image_index: {validation_image_index},", end=' ')
	if detection_rate == 1: # label과 같은 수의 object를 detection했을 때
		success_detection_num +=1
		#print(f" detection_rate = {detection_rate}")

		# detection_rate == 100% 일 때 correct_answers_class_num 계산 
		for each_object_num in range(object_num): 
			
			confidence_bounding_box = confidence_bounding_box_list[each_object_num]
			# compute x, y center coordinate 
			xcenter = int((confidence_bounding_box['left'] + confidence_bounding_box['right'] - 1.0) /2) # 1.0은 int()감안
			ycenter = int((confidence_bounding_box['top'] + confidence_bounding_box['bottom'] - 1.0) /2) 
			
			pred_list[each_object_num][0] = xcenter
			pred_list[each_object_num][1] = ycenter
			# pred_class_num  cat이면 7.0 반환
			pred_list[each_object_num][2] = class_name_to_label_dict[str(confidence_bounding_box_list[0]['class_name'])] 

		if object_num == 1:
			# label class와 예측한 class가 같다면
			index_one = tf.argmax(labels[0][4], axis = 0)
			if int(index_one) == num_classes:
				correct_answers_class_num +=1
		else:  # image에 object가 2개 이상일 때
			x_center_sort_labels = x_y_center_sort(labels, "x") # x좌표 기준으로 정렬한 labels
			y_center_sort_labels = x_y_center_sort(labels, "y") # y좌표 기준으로 정렬한 labels
			x_center_sort_pred_list = x_y_center_sort(pred_list, "x")  	# x좌표 기준으로 정렬한 pred_list
			y_center_sort_pred_list = x_y_center_sort(pred_list, "y")	# y좌표 기준으로 정렬한 pred_list

			# x좌표가 낮은 위치의 image부터 큰 위치의 image까지 detected image의 class가 동일지 확인
			for x_each_object_num in range(object_num): 
				x_center_sort_label = x_center_sort_labels[x_each_object_num, :]   	# one_hot_label_num
				x_center_sort_pred = x_center_sort_pred_list[x_each_object_num, :]	# pred_class_num
				
				# one_hot 형태의 label num을 class num형태로 변환
				x_o_h_index = tf.argmax(x_center_sort_label[4]) 
				x_label_class_num = int(class_name_to_label_dict[str(confidence_bounding_box_list[x_o_h_index]['class_name'])])
				
				x_pred_class_num = int(x_center_sort_pred[2])
				if x_label_class_num == x_pred_class_num : # class가 동일하면 pass
					pass
				else : 
					break # 하나라도 다르면 break

				if x_each_object_num == object_num-1: # x좌표 기준으로 위 조건이 만족한다면
					# y좌표가 낮은 위치의 image부터 큰 위치의 image까지 detected image의 calss가 동일지 확인
					for y_each_object_num in range(object_num):
						y_center_sort_label = y_center_sort_labels[y_each_object_num, :]
						y_center_sort_pred = y_center_sort_pred_list[y_each_object_num, :]

						y_o_h_index = tf.argmax(y_center_sort_label[4]) 
						y_label_class_num = int(class_name_to_label_dict[str(confidence_bounding_box_list[y_o_h_index]['class_name'])])
						
						y_pred_class_num = int(y_center_sort_pred[2])
						if y_label_class_num == y_pred_class_num:
							pass
						else : 
							break # 하나라도 다르면 break	

						if x_each_object_num == object_num-1:   # y좌표 기준으로도 위 조건이 만족한다면 
							correct_answers_class_num +=1
	elif detection_rate > 1: # label보다 더 많은 object를 detection했을 때
		#print("Over detection")
		detection_rate = 0.0
	else :
		#print(f"detection_rate = {detection_rate}")
		pass

		
	return success_detection_num, correct_answers_class_num, detection_rate
```



**performance_evaluation**

성능 평가는 세 가지 경우를 고려했습니다.

- **average_detection_rate**

  result에 대한 average object detection rate입니다.

  특정 조건을 만족하는 Bbox가 존재하는 경우에 대한 비율을 계산합니다. 

  Performance Evaluation Index 중 **Recall**의 방법을 따랐습니다.
  $$
  Detection\ Rate = \frac{Num\ Detected\ Object}{Num\ Label\ Object} * 100%
  $$

  $$
  Average\ Detection\ Rate = \frac{Sum \ Detection\ Rate }{Num\ Test\ Image}
  $$

  

- **perfect_detection_accuracy**

  object detection이 이루어진 result중 완벽한 object detection이 이루어진 비율입니다.
  $$
  Perfect\ Detection\ Accuracy = \frac{Num\ Perfect\ Detection }{Num\ Test\ Image}
  $$
  

  > label object가 1개일 때 2개 이상을 감지하면 over detection
  >
  > label object가 2개일 때 1개만을 감지하면 low detection
  >
  > label object가 n개일 때 n를 감지하면 perfect detection

  위의 detection_rate == 100% 인 경우 perfect detection인 것으로 결정했습니다.

- **classification_accuracy**

  result에 대한 대한 정확한 classification이 이루어진 비율입니다.

  perfect detection이라는 전제 조건에서 성공적인 classification가 이루어졌는지 확인합니다. (즉, perfect detection인 경우가 아니면 success classification 확인 과정을 수행하지 않았습니다.)
  $$
  Classification Accuracy = \frac{Num\ Correct\ Answers\ Class }{Num\ Test\ Image}
  $$
  

  *success classification 확인 과정*

  1. label과 prediction의 object list를 x좌표 기준으로 올림차순 정렬을 수행한다.

  2. x좌표가 낮은 object부터 x좌표가 높은 object 순으로 label과 prediction의 class name이 동일한지 확인한다.
  3. 2번의 조건이 만족하면, label과 prediction의 object list를 y좌표 기준으로 올림차순 정렬을 수행한다.
  4. y좌표가 낮은 object부터 y좌표가 높은 object 순으로 label과 prediction의 class name이 동일한지 확인한다.
  5. 1, 2, 3, 4번의 동작에서 모든 조건에 부합한 경우라면, success classification인 것으로 간주한다.





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

	return (checkpoint_path,
			train_summary_writer,
			validation_summary_writer)


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
		ckpt_manager.save(checkpoint_number=ckpt.step)  
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
    

def find_confidence_bounding_box(bounding_box_info_list, confidence_threshold):
	bounding_box_info_list_sorted = sorted(bounding_box_info_list,
											key=itemgetter('iou'),
											reverse=True)
	confidence_bounding_box_list = list()
	check = False

	# confidence값이 confidence_threshold 이상인 Bbox는 모두 표현
	for index in range(len(bounding_box_info_list_sorted)):
		if (bounding_box_info_list_sorted[index]['iou'] > confidence_threshold
			and bounding_box_info_list_sorted[index]['confidence_score'] > 0.6):
			confidence_bounding_box_list.append(bounding_box_info_list_sorted[index])
			# print(f"confidence_score : {bounding_box_info_list_sorted[index]['iou']:.2f}")
			check = True
		else : 
			break

	return confidence_bounding_box_list, check

def yolo_format_to_bounding_box_dict(xcenter, ycenter, box_w, box_h, class_name, confidence_score, iou):
	# the zero coordinate of image located
	bounding_box_info = dict()
	bounding_box_info['left'] = int(xcenter - (box_w / 2))
	bounding_box_info['top'] = int(ycenter + (box_h / 2))
	bounding_box_info['right'] = int(xcenter + (box_w / 2))
	bounding_box_info['bottom'] = int(ycenter - (box_h / 2))
	bounding_box_info['class_name'] = class_name
	bounding_box_info['confidence_score'] = confidence_score
	bounding_box_info['iou'] = iou

	return bounding_box_info


def iou(yolo_pred_boxes, ground_truth_boxes):

	boxes1 = yolo_pred_boxes
	boxes2 = ground_truth_boxes

	# yolo_pred_boxes의 0:xcenter, 1:ycenter, 2:w, 3:h
	# x-(w/2) == Bbox의 좌측 하단 꼭지점, y+(h/2) == Bbox의 우측 상단 꼭지점
	# boxes1에는 [xmin, ymin, xmax, ymax] 사각형 꼭지점의 x, y좌표 할당
	boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
					   boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
	boxes1 = tf.transpose(boxes1, [1, 2, 3, 0]) 	# shape을 [4 7 7 2] 에서 [7 7 2 4]로 변경

	
	# ground_truth_boxes의 0:xcenter, 1:ycenter, 2:w, 3:h
	boxes2 = tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
					   boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])
	boxes2 = tf.cast(boxes2, tf.float32)


	# calculate the left up point
	lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])  # 교집합 영역의 좌측 하단 꼭지점 좌표
	rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])	# 교집합 영역의 우측 상단 꼭지점 좌표

	# intersectiony
	intersection = rd - lu  

	inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1] # x크기 * y크기 == 넓이

	# 위에서 계산된 intersection 영역 중에서 x, y 모두 0보다 큰 값이 있어야 교집합이 존재.
	mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)

	# 각 cell 마다, 그리고 각 Bbox중 교집합이 존재하는 cell과 box만 True
	inter_square = mask * inter_square

	# calculate the boxs1 square and boxs2 square
	square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1]) # yolo_pred_boxes의 넓이
	square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1]) 									# ground_truth_boxes의 넓이

	iou = inter_square / (square1 + square2 - inter_square + 1e-6)
	# '1e-6' : for the denominator to not zero
	# 교집합 영역이 없으면 iou는 zero
	return iou


def generate_color(num_classes):
	hsv_tuples = [(x / num_classes, 1., 1.)
				   for x in range(num_classes)]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(
		map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

	np.random.seed(10101)  		# Fixed seed for consistent colors across runs.
	np.random.shuffle(colors)  	# Shuffle colors to decorrelate adjacent classes.
	np.random.seed(None) 		# Reset seed to default.

	return colors

# detect할 class에 대한 label만 추려내고, 나머지 label은 0으로 만드는 function
def remove_irrelevant_label(batch_bbox, batch_labels, class_name_dict):
	tmp_labels = np.zeros_like(batch_labels)
	tmp_bbox = np.zeros_like(batch_bbox)

	for i in range(int(tf.shape(batch_labels)[0])): 		# image 1개당
		for j in range(int(tf.shape(batch_labels)[1])):		# object 1개당
			for lable_num in class_name_dict.keys(): 
				if batch_labels[i][j] == lable_num:
					tmp_labels[i][j] = batch_labels[i][j]
					tmp_bbox[i][j] = batch_bbox[i][j]
					continue

	batch_labels = tf.constant(tmp_labels)
	batch_bbox = tf.constant(tmp_bbox)

	return batch_bbox, batch_labels

def x_y_center_sort(labels, taget):

	tmp = np.zeros_like(labels)
	if taget == "x":
		label = list(np.array(labels[:, 0]))
	elif taget == "y":
		label = list(np.array(labels[:, 1]))

	origin_label = label.copy()
	label.sort()
	for i_index, i_value in enumerate(label):
		for j_index, j_value in enumerate(origin_label):
			if i_value == j_value:
				tmp[i_index] = labels[j_index]
				continue
	labels = tmp
	
	return labels

def performance_evaluation(confidence_bounding_box_list,
						   object_num,
						   labels,
						   class_name_to_label_dict,
						   validation_image_index,
						   num_classes):

	x_center_sort_labels = None
	y_center_sort_labels = None
	x_center_sort_pred = None
	y_center_sort_pred = None

	pred_list = np.zeros(shape =(object_num, 3))

	correct_answers_class_num = 0.0  # classification accuracy 계산을 위한 값
	success_detection_num = 0.0 # perfect detection accuracy 계산을 위한 값

	# label object 중 detection한 object의 비율
	detection_rate = len(confidence_bounding_box_list)/object_num  
	#print(f"image_index: {validation_image_index},", end=' ')
	if detection_rate == 1: # label과 같은 수의 object를 detection했을 때
		success_detection_num +=1
		#print(f" detection_rate = {detection_rate}")

		# detection_rate == 100% 일 때 correct_answers_class_num 계산 
		for each_object_num in range(object_num): 
			
			confidence_bounding_box = confidence_bounding_box_list[each_object_num]
			# compute x, y center coordinate 
			xcenter = int((confidence_bounding_box['left'] + confidence_bounding_box['right'] - 1.0) /2) # 1.0은 int()감안
			ycenter = int((confidence_bounding_box['top'] + confidence_bounding_box['bottom'] - 1.0) /2) 
			
			pred_list[each_object_num][0] = xcenter
			pred_list[each_object_num][1] = ycenter
			# pred_class_num  cat이면 7.0 반환
			pred_list[each_object_num][2] = class_name_to_label_dict[str(confidence_bounding_box_list[0]['class_name'])] 

		if object_num == 1:
			# label class와 예측한 class가 같다면
			index_one = tf.argmax(labels[0][4], axis = 0)
			if int(index_one) == num_classes:
				correct_answers_class_num +=1
		else:  # image에 object가 2개 이상일 때
			x_center_sort_labels = x_y_center_sort(labels, "x") # x좌표 기준으로 정렬한 labels
			y_center_sort_labels = x_y_center_sort(labels, "y") # y좌표 기준으로 정렬한 labels
			x_center_sort_pred_list = x_y_center_sort(pred_list, "x")  	# x좌표 기준으로 정렬한 pred_list
			y_center_sort_pred_list = x_y_center_sort(pred_list, "y")	# y좌표 기준으로 정렬한 pred_list

			# x좌표가 낮은 위치의 image부터 큰 위치의 image까지 detected image의 class가 동일지 확인
			for x_each_object_num in range(object_num): 
				x_center_sort_label = x_center_sort_labels[x_each_object_num, :]   	# one_hot_label_num
				x_center_sort_pred = x_center_sort_pred_list[x_each_object_num, :]	# pred_class_num
				
				# one_hot 형태의 label num을 class num형태로 변환
				x_o_h_index = tf.argmax(x_center_sort_label[4]) 
				x_label_class_num = int(class_name_to_label_dict[str(confidence_bounding_box_list[x_o_h_index]['class_name'])])
				
				x_pred_class_num = int(x_center_sort_pred[2])
				if x_label_class_num == x_pred_class_num : # class가 동일하면 pass
					pass
				else : 
					break # 하나라도 다르면 break

				if x_each_object_num == object_num-1: # x좌표 기준으로 위 조건이 만족한다면
					# y좌표가 낮은 위치의 image부터 큰 위치의 image까지 detected image의 calss가 동일지 확인
					for y_each_object_num in range(object_num):
						y_center_sort_label = y_center_sort_labels[y_each_object_num, :]
						y_center_sort_pred = y_center_sort_pred_list[y_each_object_num, :]

						y_o_h_index = tf.argmax(y_center_sort_label[4]) 
						y_label_class_num = int(class_name_to_label_dict[str(confidence_bounding_box_list[y_o_h_index]['class_name'])])
						
						y_pred_class_num = int(y_center_sort_pred[2])
						if y_label_class_num == y_pred_class_num:
							pass
						else : 
							break # 하나라도 다르면 break	

						if x_each_object_num == object_num-1:   # y좌표 기준으로도 위 조건이 만족한다면 
							correct_answers_class_num +=1
	elif detection_rate > 1: # label보다 더 많은 object를 detection했을 때
		#print("Over detection")
		detection_rate = 0.0
	else :
		#print(f"detection_rate = {detection_rate}")
		pass

		
	return success_detection_num, correct_answers_class_num, detection_rate
```

