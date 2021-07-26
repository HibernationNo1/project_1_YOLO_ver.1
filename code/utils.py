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
	average_detection_rate_writer = tf.summary.create_file_writer(tensorboard_log_path +  '/average_detection_rate')
	perfect_detection_accuracy_writer = tf.summary.create_file_writer(tensorboard_log_path +  '/perfect_detection_accuracy')
	classification_accuracy_writer = tf.summary.create_file_writer(tensorboard_log_path +  '/classification_accuracy')  
	
	
	# pass if the path exist. or not, create directory on path
	if not os.path.isdir(model_path):
		os.makedirs(model_path, exist_ok=True)
		os.mkdir(checkpoint_path)


	return (checkpoint_path,
			train_summary_writer,
			validation_summary_writer,
			average_detection_rate_writer, 
			perfect_detection_accuracy_writer,
			classification_accuracy_writer)


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
    

def find_enough_confidence_bounding_box(bounding_box_info_list, tensorboard_log_path, step, validation_image_index):
	bounding_box_info_list_sorted = sorted(bounding_box_info_list,
											key=itemgetter('confidence_score'),
											reverse=True)
	confidence_bounding_box_list = list()

	# 가장 큰 confidence_score를 저장
	print(f'image index:{validation_image_index},  confidence_score: {bounding_box_info_list_sorted[0]["confidence_score"]}')
	
	# confidence값이 0.5 이상인 Bbox는 모두 표현
	for index, features in enumerate(bounding_box_info_list_sorted):
		if bounding_box_info_list_sorted[index]['confidence_score'] > 0.5:
			confidence_bounding_box_list.append(bounding_box_info_list_sorted[index])

	return confidence_bounding_box_list

def yolo_format_to_bounding_box_dict(xcenter, ycenter, box_w, box_h, class_name, confidence_score):
	# the zero coordinate of image located
	bounding_box_info = dict()
	bounding_box_info['left'] = int(xcenter - (box_w / 2))
	bounding_box_info['top'] = int(ycenter + (box_h / 2))
	bounding_box_info['right'] = int(xcenter + (box_w / 2))
	bounding_box_info['bottom'] = int(ycenter - (box_h / 2))
	bounding_box_info['class_name'] = class_name
	bounding_box_info['confidence_score'] = confidence_score

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
	lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])  # 교집합 영역의 우측 상단 꼭지점 좌표
	rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])	# 교집합 영역의 좌측 하단 꼭지점 좌표

	# intersectiony
	intersection = rd - lu  

	inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1] # x크기 * y크기 == 넓역

	# 위에서 계산된 intersection 영역 중에서 0보다 큰 영역만이 진짜 교집합 영역(음수 * 음수)
	mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)

	# 각 cell 마다, 그리고 각 Bbox마다 교집합 영역 계산
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
def remove_irrelevant_label(batch_labels, class_name_dict):
	tmp = np.zeros_like(batch_labels)

	for i in range(int(tf.shape(batch_labels)[0])):
		for j in range(int(tf.shape(batch_labels)[1])):
			for lable_num in class_name_dict.keys(): 
				if batch_labels[i][j] == lable_num:
					tmp[i][j] = batch_labels[i][j]
					continue
	batch_labels = tf.constant(tmp)

	return batch_labels

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
	labels = tf.constant(tmp)
	
	return labels

def performance_evaluation(confidence_bounding_box_list,
						   object_num,
						   labels,
						   class_name_to_label_dict,
						   validation_image_index):

	x_center_sort_labels = None
	y_center_sort_labels = None
	x_center_sort_pred = None
	y_center_sort_pred = None

	pred_list = np.zeros(shape =(object_num, 3))

	correct_answers_class_num = 0.0  # classification accuracy 계산을 위한 값
	success_detection_num = 0.0 # perfect detection accuracy 계산을 위한 값

	# label object 중 detection한 object의 비율
	detection_rate = len(confidence_bounding_box_list)/object_num  
	print(f"image_index: {validation_image_index},", end=' ')
	if detection_rate == 1: # label과 같은 수의 object를 detection했을 때
		success_detection_num +=1
		print(f" detection_rate = {detection_rate}")

		# detection_rate == 100% 일 때 correct_answers_class_num 계산 
		for each_object_num in range(object_num): 
			
			confidence_bounding_box = confidence_bounding_box_list[each_object_num]
			# compute x, y center coordinate 
			xcenter = int((confidence_bounding_box['left'] + confidence_bounding_box['right'] - 1.0) /2) # 1.0은 int()감안
			ycenter = int((confidence_bounding_box['top'] + confidence_bounding_box['bottom'] - 1.0) /2) 
			
			pred_list[each_object_num][0] = xcenter
			pred_list[each_object_num][1] = ycenter
			pred_list[each_object_num][2] = class_name_to_label_dict[str(confidence_bounding_box_list[0]['class_name'])] # pred_class_num

		if object_num == 1:
			# label class와 예측한 class가 같다면
			if int(labels[0][4]) == class_name_to_label_dict[str(confidence_bounding_box_list[0]['class_name'])]:
				correct_answers_class_num +=1
		else:  # image에 object가 2개 이상일 때
			x_center_sort_labels = x_y_center_sort(labels, "x") # x좌표 기준으로 정렬한 labels
			y_center_sort_labels = x_y_center_sort(labels, "y") # y좌표 기준으로 정렬한 labels
			x_center_sort_pred_list = x_y_center_sort(pred_list, "x")  	# x좌표 기준으로 정렬한 pred_list
			y_center_sort_pred_list = x_y_center_sort(pred_list, "y")	# y좌표 기준으로 정렬한 pred_list

			# x좌표가 낮은 위치의 image부터 큰 위치의 image까지 detected image의 class가 동일지 확인
			for x_each_object_num in range(object_num): 
				x_center_sort_label = x_center_sort_labels[x_each_object_num, :]
				x_center_sort_pred = x_center_sort_pred_list[x_each_object_num, :]
				
				if int(x_center_sort_label[4]) == int(x_center_sort_pred[2]): # class가 동일하면 pass
					pass
				else : 
					break # 하나라도 다르면 break

				if x_each_object_num == object_num-1: # x좌표 기준으로 위 조건이 만족한다면
					# y좌표가 낮은 위치의 image부터 큰 위치의 image까지 detected image의 calss가 동일지 확인
					for y_each_object_num in range(object_num):
						y_center_sort_label = y_center_sort_labels[y_each_object_num, :]
						y_center_sort_pred = y_center_sort_pred_list[y_each_object_num, :]
						if int(y_center_sort_label[4]) == int(y_center_sort_pred[2]):
							pass
						else : 
							break # 하나라도 다르면 break	

						if x_each_object_num == object_num-1:   # y좌표 기준으로도 위 조건이 만족한다면 
							correct_answers_class_num +=1
	elif detection_rate > 1: # label보다 더 많은 object를 detection했을 때
		print("Over detection")
	else :
		print(f"detection_rate = {detection_rate}")

		
	return success_detection_num, correct_answers_class_num, detection_rate

