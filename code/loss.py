import tensorflow as tf
import numpy as np
from utils import iou

from dataset import class_name_dict
# class_name_dict = { 7: "cat", 9:"cow" }
def class_loss_one_hot(num_classes):
	index_list = [i for i in range(num_classes)]
	P_one_hot = (tf.one_hot(tf.cast((index_list), tf.int32), num_classes, dtype=tf.float32))
	return P_one_hot


def yolo_loss(predict,
			  labels,
			  each_object_num,
			  num_classes,
			  boxes_per_cell,
			  cell_size,
			  input_width,
			  input_height,
			  coord_scale,
			  object_scale,
			  noobject_scale,
			  class_scale,
			  P_one_hot,
			  class_loss_object):

	# parse only coordinate vector
	# predict의 shape [tf.shape(predict)[0], cell_size, cell_size, num_classes + 5 * boxes_per_cell]
	predict_boxes = predict[:, :, num_classes + boxes_per_cell:]
	predict_boxes = tf.reshape(predict_boxes, [cell_size, cell_size, boxes_per_cell, 4])

	# prediction : absolute coordinate
	pred_xcenter = predict_boxes[:, :, :, 0]
	pred_ycenter = predict_boxes[:, :, :, 1]
	pred_sqrt_w = tf.sqrt(tf.minimum(input_width * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
	pred_sqrt_h = tf.sqrt(tf.minimum(input_height * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))
	pred_sqrt_w = tf.cast(pred_sqrt_w, tf.float32)
	pred_sqrt_h = tf.cast(pred_sqrt_h, tf.float32)

	# parse labe
	labels = np.array(labels) 
	labels = labels.astype('float32')
	label = labels[each_object_num, :]

	xcenter = label[0]
	ycenter = label[1]
	sqrt_w = tf.sqrt(label[2])
	sqrt_h = tf.sqrt(label[3])

	# calulate iou between ground-truth and predictions
	# 각 cell의 각 Bbox와 label과의 iou계산 tf.shape(iou_predict_truth):, [7 7 2]
	iou_predict_truth = iou(predict_boxes, label[0:4]) 

	# find best box mask
	I = iou_predict_truth
	max_I = tf.reduce_max(I, 2, keepdims=True)
	best_box_mask = tf.cast((I >= max_I), tf.float32)

	# set object_loss information(confidence, object가 있을 확률)
	C = iou_predict_truth
	pred_C = predict[:, :, num_classes:num_classes + boxes_per_cell] 


	# set class_loss information(probability, 특정 class일 확률)
	P = 0.0
	for i in range(num_classes):
		if label[4] == list(class_name_dict.keys())[i]:
			P = P_one_hot[i]

	pred_P = predict[:, :, 0:num_classes] 
	temp_pred_P = np.zeros_like(pred_P)
	for i in range(cell_size):
			for j in range(cell_size):
				temp_pred_P[i][j] = tf.nn.softmax(pred_P[i][j]) 
	pred_P = tf.constant(temp_pred_P)

	# find object exists cell mask
	object_exists_cell = np.zeros([cell_size, cell_size, 1])
	object_exists_cell_i, object_exists_cell_j = int(cell_size * ycenter / input_height), int(cell_size * xcenter / input_width)
	object_exists_cell[object_exists_cell_i][object_exists_cell_j] = 1

	# set coord_loss
	coord_loss = (tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_xcenter - xcenter) / (input_width / cell_size)) +
					tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_ycenter - ycenter) / (input_height / cell_size)) +
					tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_w - sqrt_w)) / input_width +
					tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_h - sqrt_h)) / input_height ) \
				* coord_scale

	# object_loss
	object_loss = tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_C - C)) * object_scale

	# noobject_loss
	noobject_loss = tf.nn.l2_loss((1 - object_exists_cell) * (pred_C)) * noobject_scale

	# class loss
	class_loss = class_loss_object(P, pred_P)

	# sum every loss
	total_loss = coord_loss + object_loss + noobject_loss + class_loss
        
	return total_loss, coord_loss, object_loss, noobject_loss, class_loss
