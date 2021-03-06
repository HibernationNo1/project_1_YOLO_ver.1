import cv2
import os
import shutil
import sys
import numpy as np
import tensorflow as tf
import colorsys
from operator import itemgetter

import os
from tmp import YOLOv1


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

	# confidence?????? confidence_threshold ????????? Bbox??? ?????? ??????
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

	# yolo_pred_boxes??? 0:xcenter, 1:ycenter, 2:w, 3:h
	# x-(w/2) == Bbox??? ?????? ?????? ?????????, y+(h/2) == Bbox??? ?????? ?????? ?????????
	# boxes1?????? [xmin, ymin, xmax, ymax] ????????? ???????????? x, y?????? ??????
	boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
					   boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
	boxes1 = tf.transpose(boxes1, [1, 2, 3, 0]) 	# shape??? [4 7 7 2] ?????? [7 7 2 4]??? ??????

	
	# ground_truth_boxes??? 0:xcenter, 1:ycenter, 2:w, 3:h
	boxes2 = tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
					   boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])
	boxes2 = tf.cast(boxes2, tf.float32)


	# calculate the left up point
	lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])  # ????????? ????????? ?????? ?????? ????????? ??????
	rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])	# ????????? ????????? ?????? ?????? ????????? ??????

	# intersectiony
	intersection = rd - lu  

	inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1] # x?????? * y?????? == ??????

	# ????????? ????????? intersection ?????? ????????? x, y ?????? 0?????? ??? ?????? ????????? ???????????? ??????.
	mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)

	# ??? cell ??????, ????????? ??? Bbox??? ???????????? ???????????? cell??? box??? True
	inter_square = mask * inter_square

	# calculate the boxs1 square and boxs2 square
	square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1]) # yolo_pred_boxes??? ??????
	square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1]) 									# ground_truth_boxes??? ??????

	iou = inter_square / (square1 + square2 - inter_square + 1e-6)
	# '1e-6' : for the denominator to not zero
	# ????????? ????????? ????????? iou??? zero
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

# detect??? class??? ?????? label??? ????????????, ????????? label??? 0?????? ????????? function
def remove_irrelevant_label(batch_bbox, batch_labels, class_name_dict):
	tmp_labels = np.zeros_like(batch_labels)
	tmp_bbox = np.zeros_like(batch_bbox)

	for i in range(int(tf.shape(batch_labels)[0])): 		# image 1??????
		for j in range(int(tf.shape(batch_labels)[1])):		# object 1??????
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

	correct_answers_class_num = 0.0  # classification accuracy ????????? ?????? ???
	success_detection_num = 0.0 # perfect detection accuracy ????????? ?????? ???

	# label object ??? detection??? object??? ??????
	detection_rate = len(confidence_bounding_box_list)/object_num  
	#print(f"image_index: {validation_image_index},", end=' ')
	if detection_rate == 1: # label??? ?????? ?????? object??? detection?????? ???
		success_detection_num +=1
		#print(f" detection_rate = {detection_rate}")

		# detection_rate == 100% ??? ??? correct_answers_class_num ?????? 
		for each_object_num in range(object_num): 
			
			confidence_bounding_box = confidence_bounding_box_list[each_object_num]
			# compute x, y center coordinate 
			xcenter = int((confidence_bounding_box['left'] + confidence_bounding_box['right'] - 1.0) /2) # 1.0??? int()??????
			ycenter = int((confidence_bounding_box['top'] + confidence_bounding_box['bottom'] - 1.0) /2) 
			
			pred_list[each_object_num][0] = xcenter
			pred_list[each_object_num][1] = ycenter
			# pred_class_num  cat?????? 7.0 ??????
			pred_list[each_object_num][2] = class_name_to_label_dict[str(confidence_bounding_box_list[0]['class_name'])] 

		if object_num == 1:
			# label class??? ????????? class??? ?????????
			index_one = tf.argmax(labels[0][4], axis = 0)
			if int(index_one) == num_classes:
				correct_answers_class_num +=1
		else:  # image??? object??? 2??? ????????? ???
			x_center_sort_labels = x_y_center_sort(labels, "x") # x?????? ???????????? ????????? labels
			y_center_sort_labels = x_y_center_sort(labels, "y") # y?????? ???????????? ????????? labels
			x_center_sort_pred_list = x_y_center_sort(pred_list, "x")  	# x?????? ???????????? ????????? pred_list
			y_center_sort_pred_list = x_y_center_sort(pred_list, "y")	# y?????? ???????????? ????????? pred_list

			# x????????? ?????? ????????? image?????? ??? ????????? image?????? detected image??? class??? ????????? ??????
			for x_each_object_num in range(object_num): 
				x_center_sort_label = x_center_sort_labels[x_each_object_num, :]   	# one_hot_label_num
				x_center_sort_pred = x_center_sort_pred_list[x_each_object_num, :]	# pred_class_num
				
				# one_hot ????????? label num??? class num????????? ??????
				x_o_h_index = tf.argmax(x_center_sort_label[4]) 
				x_label_class_num = int(class_name_to_label_dict[str(confidence_bounding_box_list[x_o_h_index]['class_name'])])
				
				x_pred_class_num = int(x_center_sort_pred[2])
				if x_label_class_num == x_pred_class_num : # class??? ???????????? pass
					pass
				else : 
					break # ???????????? ????????? break

				if x_each_object_num == object_num-1: # x?????? ???????????? ??? ????????? ???????????????
					# y????????? ?????? ????????? image?????? ??? ????????? image?????? detected image??? calss??? ????????? ??????
					for y_each_object_num in range(object_num):
						y_center_sort_label = y_center_sort_labels[y_each_object_num, :]
						y_center_sort_pred = y_center_sort_pred_list[y_each_object_num, :]

						y_o_h_index = tf.argmax(y_center_sort_label[4]) 
						y_label_class_num = int(class_name_to_label_dict[str(confidence_bounding_box_list[y_o_h_index]['class_name'])])
						
						y_pred_class_num = int(y_center_sort_pred[2])
						if y_label_class_num == y_pred_class_num:
							pass
						else : 
							break # ???????????? ????????? break	

						if x_each_object_num == object_num-1:   # y?????? ??????????????? ??? ????????? ??????????????? 
							correct_answers_class_num +=1
	elif detection_rate > 1: # label?????? ??? ?????? object??? detection?????? ???
		#print("Over detection")
		detection_rate = 0.0
	else :
		#print(f"detection_rate = {detection_rate}")
		pass

		
	return success_detection_num, correct_answers_class_num, detection_rate

