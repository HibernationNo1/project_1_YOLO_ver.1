import tensorflow as tf
import numpy as np
import cv2
import os

from absl import flags
from absl import app

from model import YOLOv1
from dataset import load_pascal_voc_dataset_for_test, process_each_ground_truth
from utils import (draw_bounding_box_and_label_info, 
				   generate_color,
				   find_enough_confidence_bounding_box, 
				   yolo_format_to_bounding_box_dict)
from utils import iou


flags.DEFINE_string('cp_path', default='saved_model', help='path to a directory to restore checkpoint file')
flags.DEFINE_string('test_dir', default='test_result', help='directory which test prediction result saved')

FLAGS = flags.FLAGS

# set voc label dictionary
label_to_class_dict = {
	0:"cat", 1: "cow"
}
cat_class_to_label_dict = {v: k for k, v in label_to_class_dict.items()}

from dataset import class_name_dict  
# class_name_dict = { 7: "cat", 9:"cow" }

# set configuration value
batch_size = 1
input_width = 224 	# original paper : 448
input_height = 224 	# original paper : 448
cell_size = 7
num_classes = 2 	# original paper : 20
boxes_per_cell = 2

# set color_list for drawing
color_list = generate_color(num_classes)

test_data = load_pascal_voc_dataset_for_test(batch_size)


def main(_):
	# check if checkpoint path exists
	from train import dir_name
	checkpoint_path = os.path.join(os.getcwd(), dir_name , FLAGS.cp_path)
	if not os.path.exists(checkpoint_path):
		print('checkpoint file is not exists!')
		exit()

	# create YOLO model
	YOLOv1_model = YOLOv1(input_height, input_width, cell_size, boxes_per_cell, num_classes)

	# set checkpoint manager
	ckpt = tf.train.Checkpoint(step=tf.Variable(0), model=YOLOv1_model)
	latest_ckpt = tf.train.latest_checkpoint(checkpoint_path)

	# restore latest checkpoint
	if latest_ckpt:
		ckpt.restore(latest_ckpt)
		print('global_step : {}, checkpoint is restored!'.format(int(ckpt.step)))

	num_images = len(list(test_data))  # batch_size = 1
	print('total test image :', num_images)
	for image_num, features in enumerate(test_data):
		batch_image = features['image']
		batch_bbox = features['objects']['bbox']
		batch_labels = features['objects']['label']

		batch_image = tf.squeeze(batch_image, axis=1)
		batch_bbox = tf.squeeze(batch_bbox, axis=1)
		batch_labels = tf.squeeze(batch_labels, axis=1)

		image, labels, object_num = process_each_ground_truth(batch_image[0], batch_bbox[0], batch_labels[0], input_width, input_height)

		drawing_image = image
		image = tf.expand_dims(image, axis=0)

		predict = YOLOv1_model(image)
		predict = tf.reshape(predict, 
                             [tf.shape(predict)[0], cell_size, cell_size, num_classes + 5 * boxes_per_cell])  # 7x7x(20+5*2) = 1470 -> 7x7x30

		predict_boxes = predict[0, :, :, num_classes + boxes_per_cell:]
		predict_boxes = tf.reshape(predict_boxes, [cell_size, cell_size, boxes_per_cell, 4])


		confidence_boxes = predict[0, :, :, num_classes:num_classes + boxes_per_cell]
		confidence_boxes = tf.reshape(confidence_boxes, [cell_size, cell_size, boxes_per_cell, 1])

		class_prediction = predict[0, :, :, 0:num_classes]  
		class_prediction_value = tf.reduce_max(class_prediction, axis = 2) # for compute confidence_score
		class_prediction = tf.argmax(class_prediction, axis=2)

		confidence_score = np.zeros_like(confidence_boxes[:, :, :, 0])
		for i in range(boxes_per_cell):
			confidence_score[:, :, i] = (confidence_boxes[:, :, i, 0] * class_prediction_value)/10
		
		# make prediction bounding box list
		bounding_box_info_list = []
		for i in range(cell_size):
			for j in range(cell_size):
				for k in range(boxes_per_cell):
					pred_xcenter = predict_boxes[i][j][k][0]
					pred_ycenter = predict_boxes[i][j][k][1]
					pred_box_w = tf.minimum(input_width * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][2]))
					pred_box_h = tf.minimum(input_height * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][3]))
				   
					
					pred_class_name = label_to_class_dict[class_prediction[i][j].numpy()]                   
					pred_confidence = confidence_score[i][j][k]
					# add bounding box dict list
					bounding_box_info_list.append(
					yolo_format_to_bounding_box_dict(pred_xcenter,
													 pred_ycenter,
													 pred_box_w,
													 pred_box_h, 
													 pred_class_name, 
													 pred_confidence))

		# make ground truth bounding box list
		ground_truth_bounding_box_info_list = []
		for each_object_num in range(object_num):
			labels = np.array(labels)
			labels = labels.astype('float32')
			label = labels[each_object_num, :]
			xcenter = label[0]
			ycenter = label[1]
			box_w = label[2]
			box_h = label[3]
			class_label = label[4]

			
			# add ground-turth bounding box dict list
			for label_num in class_name_dict.keys():
				if int(class_label) == label_num:     
					ground_truth_bounding_box_info_list.append(
						yolo_format_to_bounding_box_dict(xcenter, ycenter, box_w, box_h,
						 str(class_name_dict[label_num]), 1.0))

		ground_truth_drawing_image = drawing_image.copy()
		# draw ground-truth image
		for ground_truth_bounding_box_info in ground_truth_bounding_box_info_list:
			draw_bounding_box_and_label_info(
			ground_truth_drawing_image,
			ground_truth_bounding_box_info['left'],
			ground_truth_bounding_box_info['top'],
			ground_truth_bounding_box_info['right'],
			ground_truth_bounding_box_info['bottom'],
			ground_truth_bounding_box_info['class_name'],
			ground_truth_bounding_box_info['confidence_score'],
			color_list[cat_class_to_label_dict[ground_truth_bounding_box_info['class_name']]]
			)

		# find one max confidence bounding box
		confidence_bounding_box_list = find_enough_confidence_bounding_box(bounding_box_info_list)

		# draw prediction (image 위에 bounding box 표현)
		for confidence_bounding_box in confidence_bounding_box_list:
			draw_bounding_box_and_label_info(
				drawing_image,
				confidence_bounding_box['left'],
				confidence_bounding_box['top'],
				confidence_bounding_box['right'],
				confidence_bounding_box['bottom'],
				confidence_bounding_box['class_name'],
				confidence_bounding_box['confidence_score'],
				color_list[cat_class_to_label_dict[confidence_bounding_box['class_name']]]) 



		# left : ground-truth, right : prediction
		drawing_image = np.concatenate((ground_truth_drawing_image, drawing_image), axis=1)

		# save test prediction result to png file
		if not os.path.exists(os.path.join(os.getcwd(), dir_name, FLAGS.test_dir)):
			os.mkdir(os.path.join(os.getcwd(), dir_name, FLAGS.test_dir))
		output_image_name = os.path.join(os.getcwd(), dir_name, FLAGS.test_dir, str(int(image_num)) +'_result.png')
		cv2.imwrite(output_image_name, cv2.cvtColor(drawing_image, cv2.COLOR_BGR2RGB))
		print(output_image_name + ' saved!')

if __name__ == '__main__':
	app.run(main)
