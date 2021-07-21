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
    

def find_max_confidence_bounding_box(bounding_box_info_list):
	bounding_box_info_list_sorted = sorted(bounding_box_info_list,
											key=itemgetter('confidence'),
											reverse=True)
	confidence_bounding_box_list = list()

	# confidence값이 0.5 이상인 Bbox는 모두 표현
	for index, features in enumerate(bounding_box_info_list_sorted):
		if bounding_box_info_list_sorted[index]['confidence'] > 0.1:
			confidence_bounding_box_list.append(bounding_box_info_list_sorted[index])

	return confidence_bounding_box_list

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

	# yolo_pred_boxes의 중앙 좌표  0:x, 1:y, 2:w, 3:y
	boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
					   boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
	boxes1 = tf.transpose(boxes1, [1, 2, 3, 0]) 	# shape을 [4 7 7 2] 에서 [7 7 2 4]로 변경

	# ground_truth_boxes의 중앙 좌표
	boxes2 = tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
					   boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])
	boxes2 = tf.cast(boxes2, tf.float32)
	# tf.shape(boxes1) == [4]

	# calculate the left up point
	lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])  # 두 Bbox 중 x, y의 최대값(전체 영역의 우측 상단 꼭지점 좌표)
	rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])	# 두 Bbox 중 w, h의 최소값(교집합 영역의 우측 상단 꼭지점 좌표)
  
	# intersection
	intersection = rd - lu  

	inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1] # 모든 x좌표 * y좌표

	# 위에서 계산된 intersection 영역 중에서 0인 영역만이 진짜 교집합 영역
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
