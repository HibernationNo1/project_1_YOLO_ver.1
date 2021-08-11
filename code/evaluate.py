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
				   find_confidence_bounding_box, 
				   yolo_format_to_bounding_box_dict,
				   remove_irrelevant_label,
				   performance_evaluation)
from utils import iou


flags.DEFINE_string('cp_path', default='saved_model', help='path to a directory to restore checkpoint file')
flags.DEFINE_string('test_dir', default='test_result', help='directory which test prediction result saved')

FLAGS = flags.FLAGS

from train import dir_name
# set voc label dictionary
from train import label_to_class_dict
cat_class_to_label_dict = {v: k for k, v in label_to_class_dict.items()}



from dataset import class_name_dict  
# class_name_dict = class_name_dict = { 7: "cat", 11: "dog", 12: "horse"}
class_name_to_label_dict = {v: k for k, v in class_name_dict.items()}

# set configuration valuey
batch_size = 1 	
input_width = 224 	# original paper : 448
input_height = 224 	# original paper : 448
cell_size = 7
num_classes = int(len(class_name_dict.keys())) 	# original paper : 20
boxes_per_cell = 2
confidence_threshold = 0.5
# set color_list for drawings
color_list = generate_color(num_classes)

test_data = load_pascal_voc_dataset_for_test(batch_size)
#from dataset import load_pascal_voc_dataset
#_, test_data = load_pascal_voc_dataset(batch_size)


def main(_):
	# check if checkpoint path exists
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

	detection_rate_sum = 0.0  		# average detection rate 계산을 위한 분자값
	success_detection_num = 0.0		# perfect detection accuracy 계산을 위한 분자값
	correct_answers_class_num = 0.0 # classicifiation accuracy 계산을 위한 분자값
	total_object_num = 0.0 			# classicifiation accuracy 계산을 위한 분모값
	num_images = len(list(test_data))  # batch_size = 1
	print('total test image :', num_images)

	for image_num, features in enumerate(test_data):
		batch_image = features['image']
		batch_bbox = features['objects']['bbox']
		batch_labels = features['objects']['label']

		batch_image = tf.squeeze(batch_image, axis=1)
		batch_bbox = tf.squeeze(batch_bbox, axis=1)
		batch_labels = tf.squeeze(batch_labels, axis=1)

		batch_bbox, batch_labels = remove_irrelevant_label(batch_bbox, batch_labels, class_name_dict)

		image, labels, object_num = process_each_ground_truth(batch_image[0],
															  batch_bbox[0], 
															  batch_labels[0], 
															  input_width, 
															  input_height)

		ground_truth_bounding_box_info_list = list() 	# make ground truth bounding box list
		for each_object_num in range(object_num):
			# make label image info list 
			labels = np.array(labels)
			label = labels[each_object_num, :]
			xcenter = label[0]
			ycenter = label[1]
			box_w = label[2]
			box_h = label[3]

			# [1., 0.] 일 때 index_one == 0, [0., 1.] 일 때 index_one == 1
			class_label_index = tf.argmax(label[4])
			
			# add ground-turth bounding box dict list
			# 특정 class에만 ground truth bounding box information을 draw
			for label_num in range(num_classes):
				if int(class_label_index) == label_num:     
					ground_truth_bounding_box_info_list.append(
						yolo_format_to_bounding_box_dict(xcenter, ycenter, box_w, box_h,
						 str(label_to_class_dict[label_num]), 1.0, 1.0))

		drawing_image = image
		image = tf.expand_dims(image, axis=0)

		predict = YOLOv1_model(image)

		predict_boxes = predict[2]
		predict_boxes = tf.squeeze(predict_boxes, [0])

		pred_P = tf.nn.softmax(predict[0])
		pred_P = tf.squeeze(pred_P, [0])

		class_prediction = pred_P  
		class_prediction = tf.argmax(class_prediction, axis=2)

		iou_predict_truth_list = list() 	# object_num, 7, 7, 2, 
		index_class_name_list = list() 		# object_num
		for each_object_num in range(object_num):
			iou_predict_truth_list.append(iou(predict_boxes, labels[each_object_num, 0:4]))
			index_class_name_list.append(tf.argmax(labels[each_object_num, 4]))
															  
		
		bounding_box_info_list = list()					# make prediction bounding box list
		for i in range(cell_size):
			for j in range(cell_size):
				for k in range(boxes_per_cell):
					pred_xcenter = predict_boxes[i][j][k][0]
					pred_ycenter = predict_boxes[i][j][k][1]
					pred_box_w = tf.minimum(input_width * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][2]))
					pred_box_h = tf.minimum(input_height * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][3]))

					for each_object_num in range(object_num):
						pred_class_name = label_to_class_dict[index_class_name_list[each_object_num].numpy()]
						confidence_score = pred_P[i][j][index_class_name_list[each_object_num]]
						computed_iou = iou_predict_truth_list[each_object_num][i][j][k]
				
						bounding_box_info_list.append(yolo_format_to_bounding_box_dict(pred_xcenter, 
																					   pred_ycenter,
																					   pred_box_w,
																					   pred_box_h,
																					   pred_class_name,
																					   confidence_score,
																					   computed_iou
																					   ))


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
		confidence_bounding_box_list, check = find_confidence_bounding_box(bounding_box_info_list, confidence_threshold)

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
		print(f"image index: {image_num}")

		# save test prediction result to png file
		if not os.path.exists(os.path.join(os.getcwd(), dir_name, FLAGS.test_dir)):
			os.mkdir(os.path.join(os.getcwd(), dir_name, FLAGS.test_dir))
		output_image_name = os.path.join(os.getcwd(), dir_name, FLAGS.test_dir, str(int(image_num)) +'_result.png')
		if check:
			print("Detection success")
			cv2.imwrite(output_image_name, cv2.cvtColor(drawing_image, cv2.COLOR_BGR2RGB))
			print(output_image_name + ' saved! \n')
		else : 
			print("Detection failed \n")

		detection_num, class_num, detection_rate = performance_evaluation(confidence_bounding_box_list,
																		  object_num,
																		  labels,
																		  class_name_to_label_dict,
																		  image_num,
																		  num_classes)

		detection_rate_sum +=detection_rate
		success_detection_num += detection_num
		correct_answers_class_num += class_num
		total_object_num += object_num
		

	average_detection_rate = detection_rate_sum / num_images  			# 평균 object detection 비율	
	perfect_detection_accuracy = success_detection_num / num_images		# 완벽한 object detection이 이루어진 비율
	classification_accuracy = correct_answers_class_num / num_images 	# 정확한 classicifiation이 이루어진 비율
	print(f"average_detection_rate: {average_detection_rate:.2f}")
	print(f"perfect_detection_accuracy: {perfect_detection_accuracy:.2f}")
	print(f"classification_accuracy: {classification_accuracy:.2f}")

if __name__ == '__main__':
	app.run(main)
