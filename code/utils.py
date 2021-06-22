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

	# if CONTINUE_LEARNING is True but nothing in model directory, 'CONTINUE_LEARNING' is 'False' 
	if CONTINUE_LEARNING == True and not os.path.isdir(model_path):
		CONTINUE_LEARNING = False
		print("CONTINUE_LEARNING flag has been converted to FALSE") 

	if CONTINUE_LEARNING == False and os.path.isdir(model_path):
		# left some file or information if when the training start at first
		# delete all file on model_path
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

	# pass if the path exist. or not, create directory on path
	if not os.path.isdir(model_path):
		os.makedirs(model_path, exist_ok=True)
		os.mkdir(checkpoint_path)

		# set tensorboard log
		# tensorboard_log를 write하기 위한 writer instance 만들기
		train_summary_writer = tf.summary.create_file_writer(tensorboard_log_path +  '/train')
		validation_summary_writer = tf.summary.create_file_writer(tensorboard_log_path +  '/validation')  
	# dir_name	|-- saved_model
	#			|-- tensorboard_log	|-- train
	#								|-- validation

	return checkpoint_path, train_summary_writer, validation_summary_writer

def set_checkpoint_manager(	input_height,
						   	input_width,
						   	cell_size,
				    	   	boxes_per_cell,
							num_classes,
							checkpoint_path )
	# create YOLO model
	YOLOv1_model = YOLOv1(input_height, input_width, cell_size, boxes_per_cell, num_classes)

	# set checkpoint manager
	ckpt = tf.train.Checkpoint(step=tf.Variable(0), model=YOLOv1_model)
	# step 설정은 초기 train entry point
	ckpt_manager = tf.train.CheckpointManager(ckpt,
                                            directory=checkpoint_path,
                                            max_to_keep=None)
	latest_ckpt = tf.train.latest_checkpoint(checkpoint_path)
	# latest_checkpoint : 마지막 checkpoint에서 저장된 file의 path를 return 

	# restore latest checkpoint
	# 마지막 checkpoint의 값들을 ckpt에 저장
	if latest_ckpt:
		ckpt.restore(latest_ckpt)
		print('global_step : {}, checkpoint is restored!'.format(int(ckpt.step)))
	return ckpt, ckpt_manager, YOLOv1_model


def save_tensorboard_log(train_summary_writer, optimizer, total_loss,
						 coord_loss, object_loss, noobject_loss, class_loss, ckpt):
	# 현재 시점의 step의 각 loss값을 write
	with train_summary_writer.as_default():
		tf.summary.scalar('learning_rate ', optimizer.lr(ckpt.step).numpy(), step=int(ckpt.step))
		tf.summary.scalar('total_loss', total_loss, step=int(ckpt.step))
		tf.summary.scalar('coord_loss', coord_loss, step=int(ckpt.step))
		tf.summary.scalar('object_loss ', object_loss, step=int(ckpt.step))
		tf.summary.scalar('noobject_loss ', noobject_loss, step=int(ckpt.step))
		tf.summary.scalar('class_loss ', class_loss, step=int(ckpt.step))

def save_checkpoint(ckpt, ckpt_manager, save_checkpoint_steps):
	# save checkpoint
	# ckpt.step이 FLAGS.save_checkpoint_steps에 도달 할 때마다
	if ckpt.step % save_checkpoint_steps == 0:
		ckpt_manager.save(checkpoint_number=ckpt.step)  # CheckpointManager의 parameter 저장
		print('global_step : {}, checkpoint is saved!'.format(int(ckpt.step)))

def draw_bounding_box_and_label_info(frame, x_min, y_min, x_max, y_max, label, confidence, color):
	# draw rectangle
	cv2.rectangle(
		frame,
		(x_min, y_min),
		(x_max, y_max),
		color, 3)

    # draw ladle information
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
	# 'top' = ycenter - (box_h / 2)  and 'bottom' = ycenter + (box_h / 2)
	bounding_box_info = {}
	bounding_box_info['left'] = int(xcenter - (box_w / 2))
	bounding_box_info['top'] = int(ycenter - (box_h / 2))
	bounding_box_info['right'] = int(xcenter + (box_w / 2))
	bounding_box_info['bottom'] = int(ycenter + (box_h / 2))
	bounding_box_info['class_name'] = class_name
	bounding_box_info['confidence'] = confidence

	return bounding_box_info

# 
def iou(yolo_pred_boxes, ground_truth_boxes):
	# Reference : https://github.com/nilboy/tensorflow-yolo/blob/python2.7/yolo/net/yolo_tiny_net.py#L105
	"""calculate ious
	Args:
		yolo_pred_boxes: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
		ground_truth_boxes: 1-D tensor [4] ===> (x_center, y_center, w, h)
	Return:
		iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
	"""
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
	# Reference : https://github.com/qqwweee/keras-yolo3/blob/e6598d13c703029b2686bc2eb8d5c09badf42992/yolo.py#L82
	# Generate colors for drawing bounding boxes.
	hsv_tuples = [(x / num_classes, 1., 1.)
				   for x in range(num_classes)]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(
		map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

	np.random.seed(10101)  # Fixed seed for consistent colors across runs.
	np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
	np.random.seed(None)  # Reset seed to default.

	return colors