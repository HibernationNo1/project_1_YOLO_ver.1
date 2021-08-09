import tensorflow as tf
import numpy as np
from utils import iou


def yolo_loss(predict,
			  labels,
			  each_object_num,
			  cell_size,
			  input_width,
			  input_height,
			  coord_scale,
			  object_scale,
			  noobject_scale,
			  class_scale):

	# predict[0] == pred_class 		[1, cell_size, cell_size, num_class]
	# predict[1] == pred_confidence [1, cell_size, cell_size, boxes_per_cell]
	# predict[2] == pred_coordinate	[1, cell_size, cell_size, boxes_per_cell, 4]
		
	predict_boxes = predict[2]
	predict_boxes = tf.squeeze(predict_boxes, [0])

	# prediction : absolute coordinate
	pred_xcenter = predict_boxes[:, :, :, 0]
	pred_ycenter = predict_boxes[:, :, :, 1]
	pred_sqrt_w = tf.sqrt(tf.minimum(input_width * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
	pred_sqrt_h = tf.sqrt(tf.minimum(input_height * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))
	pred_sqrt_w = tf.cast(pred_sqrt_w, tf.float32)
	pred_sqrt_h = tf.cast(pred_sqrt_h, tf.float32)

	# parse labe
	labels = np.array(labels) 
	label = labels[each_object_num, :]

	xcenter = label[0]
	ycenter = label[1] 
	sqrt_w = tf.sqrt(label[2])
	sqrt_h = tf.sqrt(label[3])


	# calulate iou between ground-truth and predictions
	# 각 cell의 각 Bbox와 label과의 iou계산 tf.shape(iou_predict_truth):, [7 7 2]
	iou_predict_truth = iou(predict_boxes, label[0:4]) 

	# find best box mask(두 Bbox중에서 IOU가 큰 Bbox선택)
	# 두 Bbox의 IOU가 같으면 첫 번째 Bbox가 best Bbox
	I = iou_predict_truth 	
	max_I = tf.reduce_max(I, 2, keepdims=True)
	best_box_mask = tf.cast((I >= max_I), tf.float32)
	
	# set object_loss information(confidence, object가 있을 확률)
	C = iou_predict_truth 

	arg_c = tf.argmax(C, axis = 2)
	tmp_object= np.zeros_like(C)
	for i in range(cell_size):
		for j in range(cell_size):
			# 특정 cell의 두 Bbox의 IOU가 0이면 해당 cell의 object label을 전부 0으로 유지
			if tf.reduce_max(C[i][j]) == 0:    	
				pass
			else :
				# 두 Bbox중 높은 iou를 가지면 1, 아니면 0의 값 (one_hot) 
				tmp_object[i][j][arg_c[i][j]] = 1  			
	C_label = tf.constant(tmp_object)


	pred_C = predict[1]
	pred_C = tf.squeeze(pred_C, [0])
	
	# set class_loss information(probability, 특정 class일 확률)
	pred_P = predict[0]
	pred_P = tf.squeeze(pred_P, [0])	
	temp_P = np.zeros_like(pred_P)
	for i in range(cell_size):
		for j in range(cell_size):
				temp_P[i][j] = label[4]
	P = tf.constant(temp_P)


	# find object exists cell mask
	object_exists_cell = np.zeros([cell_size, cell_size, 1])
	object_exists_cell_i, object_exists_cell_j = int(cell_size * ycenter / input_height), int(cell_size * xcenter / input_width)
	object_exists_cell[object_exists_cell_i][object_exists_cell_j] = 1

	coord_loss = (tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_xcenter - xcenter) / (input_width / cell_size)) +
					tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_ycenter - ycenter) / (input_height / cell_size)) +
					tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_w - sqrt_w)) / input_width +
					tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_h - sqrt_h)) / input_height ) \
				* coord_scale

	# object_loss
	object_loss =  tf.reduce_mean(object_exists_cell * best_box_mask * object_scale * tf.nn.sigmoid_cross_entropy_with_logits(C_label, pred_C))
		
	# noobject_loss
	# object loss와 noobject loss의 차이는 indicator function이다.
	noobject_loss = tf.reduce_mean((1 - object_exists_cell) * best_box_mask * tf.nn.sigmoid_cross_entropy_with_logits(C_label, pred_C) * noobject_scale)


	# class loss 
	class_loss = tf.reduce_mean(object_exists_cell * class_scale * tf.nn.softmax_cross_entropy_with_logits(P, pred_P))

	# sum every loss
	total_loss = coord_loss + object_loss + noobject_loss + class_loss
	
	
	return total_loss, coord_loss, object_loss, noobject_loss, class_loss
