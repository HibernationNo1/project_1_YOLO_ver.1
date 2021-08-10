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
batch_size = 32 	# original paper : 64
input_width = 224 	# original paper : 448
input_height = 224 	# original paper : 448
cell_size = 7
num_classes = int(len(class_name_dict.keys())) 	# original paper : 20
boxes_per_cell = 2
confidence_threshold = 0.6
# set color_list for drawings
color_list = generate_color(num_classes)

# test_data = load_pascal_voc_dataset_for_test(batch_size)
from dataset import load_pascal_voc_dataset
test_data, _ = load_pascal_voc_dataset(batch_size)

# ----------------------
coord_scale = 5	# original paper : 5  
class_scale = 0.2  	# original paper : 1
object_scale = 1	# original paper : None
noobject_scale = 0.5	# original paper : None
from loss import yolo_loss
def calculate_loss(model, batch_image, batch_bbox, batch_labels):
	total_loss = 0.0
	coord_loss = 0.0
	object_loss = 0.0
	noobject_loss = 0.0
	class_loss = 0.0

	for batch_index in range(batch_image.shape[0]): # 전체 batch에 대해서 1개씩 반복
		image, labels, object_num = process_each_ground_truth(batch_image[batch_index],
														   	  batch_bbox[batch_index],
														  	  batch_labels[batch_index],
														  	  input_width, input_height)

		# model의 inceptionV3의 input의 shape에 맞추기 위해 dumy dims 생성
		image = tf.expand_dims(image, axis=0) 

		predict = model(image)
		# predict[0] == pred_class
		# predict[1] == pred_confidence
		# predict[2] == pred_coordinate

		for object_num_index in range(object_num): # 실제 object개수만큼 for루프
            # 각 return값은 1개의 image에 대한 여러 loss 값임
			(each_object_total_loss, 
			 each_object_coord_loss, 
			 each_object_object_loss, 
			 each_object_noobject_loss, 
			 each_object_class_loss) = yolo_loss(predict,
								   				 labels,
								   				 object_num_index,
								   				 cell_size,
								   				 input_width,
												 input_height,
								   				 coord_scale,
								   				 object_scale,
								   				 noobject_scale,
								   				 class_scale)
			
            # 각각 전체의 batch에 대해서 loss 합산
			total_loss = total_loss+ each_object_total_loss
			coord_loss = coord_loss + each_object_coord_loss
			object_loss = object_loss + each_object_object_loss
			noobject_loss = noobject_loss + each_object_noobject_loss
			class_loss = class_loss + each_object_class_loss
	return total_loss, coord_loss, object_loss, noobject_loss, class_loss

def test_result(model):
	total_tset_total_loss = 0.0
	total_test_coord_loss = 0.0  
	total_test_object_loss = 0.0
	total_test_noobject_loss = 0.0  
	total_test_class_loss = 0.0

	for iter, features in enumerate(test_data):
		batch_test_image = features['image']
		batch_test_bbox = features['objects']['bbox']
		batch_test_labels = features['objects']['label']

		batch_test_image = tf.squeeze(batch_test_image, axis=1)                             
		batch_test_bbox = tf.squeeze(batch_test_bbox, axis=1)
		batch_test_labels = tf.squeeze(batch_test_labels, axis=1)

		batch_test_bbox, batch_test_labels = remove_irrelevant_label(batch_test_bbox, 
																				 batch_test_labels,
																				 class_name_dict)
	
    	# validation data와 model의 predictor간의 loss값 compute
		(test_total_loss,
		 test_coord_loss,
		 test_object_loss,
		 test_noobject_loss,
		 test_class_loss) = calculate_loss(model,
												 batch_test_image,
												 batch_test_bbox,
												 batch_test_labels)
		total_tset_total_loss = total_tset_total_loss + test_total_loss
		total_test_coord_loss = total_test_coord_loss + test_coord_loss
		total_test_object_loss = total_test_object_loss + test_object_loss
		total_test_noobject_loss = total_test_noobject_loss + test_noobject_loss
		total_test_class_loss = total_test_class_loss + test_class_loss

	print(f'total_test_total_loss : {total_tset_total_loss}')
	print(f'total_test_coord_loss : {total_test_coord_loss}')
	print(f'total_test_object_loss : {total_test_object_loss}')
	print(f'total_test_noobject_loss : {total_test_noobject_loss}')
	print(f'total_test_class_loss : {total_test_class_loss}')

# ----------------------



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

		image, labels, object_num = process_each_ground_truth(batch_image[0], batch_bbox[0], batch_labels[0], input_width, input_height)

		ground_truth_bounding_box_info_list = list() 	# make ground truth bounding box list
		# make label image info list 
		class_label_index = 0.0
		for each_object_num in range(object_num):
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

		confidence_boxes = tf.nn.sigmoid(predict[1])
		confidence_boxes = tf.squeeze(confidence_boxes, [0])

		class_prediction = tf.nn.softmax(predict[0])
		class_prediction = tf.squeeze(class_prediction, [0])
		pred_P = class_prediction
		class_prediction = tf.argmax(class_prediction, axis=2)


		bounding_box_info_list = list()					# make prediction bounding box list
		for i in range(cell_size):
			for j in range(cell_size):
				for k in range(boxes_per_cell):
					pred_xcenter = predict_boxes[i][j][k][0]
					pred_ycenter = predict_boxes[i][j][k][1]
					pred_box_w = tf.minimum(input_width * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][2]))
					pred_box_h = tf.minimum(input_height * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][3]))
					
					pred_class_name = label_to_class_dict[class_prediction[i][j].numpy()]   

					iou_predict_truth = iou(predict_boxes, label[0:4])

					# confedence_score = class_probability * intersection_of_union
					# class_probability는 여러 class중 실제 정답에 대해 예측한 확률값을 사용한다.
					confidence_score = pred_P[i][j][class_label_index] 
					computed_iou = iou_predict_truth[i][j][k]
						
					# for문이 끝나면 bounding_box_info_list에는 (object_num * cell_size * cell_size * box_per_cell)개의 bounding box의 information이 들어있다.
					# 각 bounding box의 information은 (x, y, w, h, class_name, confidence_score)이다.
					# add bounding box dict list
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
		confidence_bounding_box_list = find_confidence_bounding_box(bounding_box_info_list, confidence_threshold)

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
		test_result(YOLOv1_model)

	average_detection_rate = detection_rate_sum / num_images  			# 평균 object detection 비율	
	perfect_detection_accuracy = success_detection_num / num_images		# 완벽한 object detection이 이루어진 비율
	classification_accuracy = correct_answers_class_num / num_images 	# 정확한 classicifiation이 이루어진 비율
	print(f"average_detection_rate: {average_detection_rate:.2f}")
	print(f"perfect_detection_accuracy: {perfect_detection_accuracy:.2f}")
	print(f"classification_accuracy: {classification_accuracy:.2f}")

if __name__ == '__main__':
	app.run(main)
