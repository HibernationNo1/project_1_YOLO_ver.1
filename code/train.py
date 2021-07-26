import tensorflow as tf
import numpy as np
import random

from absl import flags
from absl import app

from tensorflow.keras.optimizers import schedules
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy

from dataset import load_pascal_voc_dataset
from loss import yolo_loss
from dataset import process_each_ground_truth
from utils import (draw_bounding_box_and_label_info,
				   find_enough_confidence_bounding_box, 
				   yolo_format_to_bounding_box_dict,
				   generate_color,
				   dir_setting,
				   save_checkpoint,
				   set_checkpoint_manager,
				   remove_irrelevant_label)

from utils import performance_evaluation

# flags instance로 hyper parameters setting
flags.DEFINE_string('checkpoint_path', default='saved_model', help='path to a directory to save model checkpoints during training')
flags.DEFINE_integer('save_checkpoint_steps', default=50, help='period at which checkpoints are saved (defaults to every 50 steps)')
flags.DEFINE_string('tensorboard_log_path', default='tensorboard_log', help='path to a directory to save tensorboard log')
flags.DEFINE_integer('validation_steps', default=50, help='period at which test prediction result and save image')  # 몇 번의 step마다 validation data로 test를 할지 결정
flags.DEFINE_integer('num_epochs', default=50, help='training epochs') # original paper : 135 epoch
flags.DEFINE_float('init_learning_rate', default=0.0001, help='initial learning rate') # original paper : 0.001 (1epoch) -> 0.01 (75epoch) -> 0.001 (30epoch) -> 0.0001 (30epoch)
flags.DEFINE_float('lr_decay_rate', default=0.75, help='decay rate for the learning rate')
flags.DEFINE_integer('lr_decay_steps', default=200, help='number of steps after which the learning rate is decayed by decay rate') # 2000번 마다 init_learning_rate * lr_decay_rate 을 실행
# 2000 step : init_learning_rate = 0.00005, 4000 step : init_learning_rate = 0.000025
flags.DEFINE_integer('num_visualize_image', default=8, help='number of visualize image for validation')
# 중간중간 validation을 할 때마다 몇 개의 batch size로 visualization을 할지 결정하는 변수

FLAGS = flags.FLAGS


# set cat label dictionary 
label_to_class_dict = {
	0: "cat", 1: "cow"
}
cat_class_to_label_dict = {v: k for k, v in label_to_class_dict.items()}

# class_name_dict을 dataset.py에서 선언하는 이유 : train.py에서 선언하면 import 순환 이슈가 발생한다.
from dataset import class_name_dict  
# class_name_dict = { 7: "cat", 9:"cow" }
class_name_to_label_dict = {v: k for k, v in class_name_dict.items()}


dir_name = 'train4'

# 이전에 했던 training을 다시 시작하거나 처음 진행할 때 False, 계속 이어서 할 땐 True 
CONTINUE_LEARNING = True


# set configuration valuey
batch_size = 32 	# original paper : 64
input_width = 224 	# original paper : 448
input_height = 224 	# original paper : 448
cell_size = 7
num_classes = int(len(class_name_dict.keys())) 	# original paper : 20
boxes_per_cell = 2

# set color_list for drawings
color_list = generate_color(num_classes)

# generate dataset
train_data, validation_data = load_pascal_voc_dataset(batch_size)

# set loss function coefficients
coord_scale = 10 	# original paper : 5  
class_scale = 0.1  	# original paper : 1
object_scale = 1	# original paper : None
noobject_scale = 0.5	# original paper : None


def calculate_loss(model, batch_image, batch_bbox, batch_labels, class_loss_object, confidence_loss_object):
	total_loss = 0.0
	coord_loss = 0.0
	object_loss = 0.0
	noobject_loss = 0.0
	class_loss = 0.0
	pred_C, pred_P = 0.0, 0.0
	for batch_index in range(batch_image.shape[0]): # 전체 batch에 대해서 1개씩 반복
		image, labels, object_num = process_each_ground_truth(batch_image[batch_index],
														   	  batch_bbox[batch_index],
														  	  batch_labels[batch_index],
														  	  input_width, input_height)
	
		image = tf.expand_dims(image, axis=0)

		predict = model(image) # 여기서 predict의 shape은 flatten vector 형태
		# flatten vector -> cell_size x cell_size x (num_classes + 5 * boxes_per_cell)
		predict = tf.reshape(predict, 
					[tf.shape(predict)[0], cell_size, cell_size, num_classes + 5 * boxes_per_cell])

		for object_num_index in range(object_num): # 실제 object개수만큼 for루프
            # 각 return값은 1개의 image에 대한 여러 loss 값임
			(each_object_total_loss, 
			 each_object_coord_loss, 
			 each_object_object_loss, 
			 each_object_noobject_loss, 
			 each_object_class_loss,
			 pred_C, pred_P) = yolo_loss(predict[0],
								   				 labels,
								   				 object_num_index,
								   				 num_classes,
								   				 boxes_per_cell,
								   				 cell_size,
								   				 input_width,
												 input_height,
								   				 coord_scale,
								   				 object_scale,
								   				 noobject_scale,
								   				 class_scale,
												 class_loss_object,
												 confidence_loss_object)
			
            # 각각 전체의 batch에 대해서 loss 합산
			total_loss = total_loss+ each_object_total_loss
			coord_loss = coord_loss + each_object_coord_loss
			object_loss = object_loss + each_object_object_loss
			noobject_loss = noobject_loss + each_object_noobject_loss
			class_loss = class_loss + each_object_class_loss
	return total_loss, coord_loss, object_loss, noobject_loss, class_loss, pred_C, pred_P


def train_step(optimizer, model, batch_image, batch_bbox, batch_labels, class_loss_object, confidence_loss_object): 
	with tf.GradientTape() as tape:
		(total_loss,
		 coord_loss,
		 object_loss,
		 noobject_loss,
		 class_loss,
		 pred_C, pred_P) = calculate_loss(model, batch_image, batch_bbox, batch_labels, class_loss_object, confidence_loss_object)
	
	gradients = tape.gradient(total_loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	return total_loss, coord_loss, object_loss, noobject_loss, class_loss

    
def save_tensorboard_log(train_summary_writer, optimizer, ckpt, 
						 total_loss, coord_loss, object_loss, noobject_loss, class_loss):
	with train_summary_writer.as_default():
		tf.summary.scalar('learning_rate ', optimizer.lr(ckpt.step).numpy(), step=int(ckpt.step))
		tf.summary.scalar('total_loss',	total_loss, step=int(ckpt.step))
		tf.summary.scalar('coord_loss', coord_loss, step=int(ckpt.step))
		tf.summary.scalar('object_loss ', object_loss, step=int(ckpt.step))
		tf.summary.scalar('noobject_loss ', noobject_loss, step=int(ckpt.step))
		tf.summary.scalar('class_loss ', class_loss, step=int(ckpt.step)) 
        
    
def save_validation_result(model,
						   ckpt, 
						   validation_summary_writer,
						   average_detection_rate_writer,
						   perfect_detection_accuracy_writer,
						   classification_accuracy_writer,
						   num_visualize_image,
						   class_loss_object,
						   confidence_loss_object):
	total_validation_total_loss = 0.0
	total_validation_coord_loss = 0.0  
	total_validation_object_loss = 0.0
	total_validation_noobject_loss = 0.0  
	total_validation_class_loss = 0.0
	pred_C, pred_P = 0.0, 0.0
	for iter, features in enumerate(validation_data):
		batch_validation_image = features['image']
		batch_validation_bbox = features['objects']['bbox']
		batch_validation_labels = features['objects']['label']

		batch_validation_image = tf.squeeze(batch_validation_image, axis=1)                             
		batch_validation_bbox = tf.squeeze(batch_validation_bbox, axis=1)
		batch_validation_labels = tf.squeeze(batch_validation_labels, axis=1)
	
    	# validation data와 model의 predictor간의 loss값 compute
		(validation_total_loss,
		 validation_coord_loss,
		 validation_object_loss,
		 validation_noobject_loss,
		 validation_class_loss,
		 pred_C, pred_P) = calculate_loss(model,
												 batch_validation_image,
												 batch_validation_bbox,
												 batch_validation_labels,
												 class_loss_object,
												 confidence_loss_object)
		total_validation_total_loss = total_validation_total_loss + validation_total_loss
		total_validation_coord_loss = total_validation_coord_loss + validation_coord_loss
		total_validation_object_loss = total_validation_object_loss + validation_object_loss
		total_validation_noobject_loss = total_validation_noobject_loss + validation_noobject_loss
		total_validation_class_loss = total_validation_class_loss + validation_class_loss
	  
	# save validation tensorboard log
	with validation_summary_writer.as_default():

		print(f'total_validation_total_loss : {total_validation_total_loss}')
		print(f'total_validation_coord_loss : {total_validation_coord_loss}')
		print(f'total_validation_object_loss : {total_validation_object_loss}')
		print(f'total_validation_noobject_loss : {total_validation_noobject_loss}')
		print(f'total_validation_class_loss : {total_validation_class_loss}')

		tf.summary.scalar('total_validation_total_loss', total_validation_total_loss, step=int(ckpt.step))
		tf.summary.scalar('total_validation_coord_loss', total_validation_coord_loss, step=int(ckpt.step))
		tf.summary.scalar('total_validation_object_loss ', total_validation_object_loss, step=int(ckpt.step))
		tf.summary.scalar('total_validation_noobject_loss ', total_validation_noobject_loss, step=int(ckpt.step))
		tf.summary.scalar('total_validation_class_loss ', total_validation_class_loss, step=int(ckpt.step))
	  
	detection_rate_sum = 0.0  		# average detection rate 계산을 위한 분자값
	success_detection_num = 0.0		# perfect detection accuracy 계산을 위한 분자값
	correct_answers_class_num = 0.0 # classicifiation accuracy 계산을 위한 분자값

	# save validation test image
	for validation_image_index in range(num_visualize_image):
		random_idx = random.randint(0, batch_validation_image.shape[0] - 1)
		image, labels, object_num = process_each_ground_truth(batch_validation_image[random_idx],
															  batch_validation_bbox[random_idx],
															  batch_validation_labels[random_idx],
															  input_width, input_height)
	
		drawing_image = image
		image = tf.expand_dims(image, axis=0)  # make dummy dimasion

		predict = model(image)
		# tf.shape(predict)[0] == batch_size
		predict = tf.reshape(predict, 
				 [tf.shape(predict)[0], cell_size, cell_size, num_classes + (5 * boxes_per_cell)])
		
		# parse prediction(x, y, w, h)
		predict_boxes = predict[0, :, :, num_classes + boxes_per_cell:]
		predict_boxes = tf.reshape(predict_boxes, [cell_size, cell_size, boxes_per_cell, 4])
		

		# pred_C : 예측한 Bbox영역 안에 object가 있을 probability
		# pred_P : 각 class에 대한 predicted probability
		# 각 셀마다 class probability가 가장 높은 prediction value의 index추출(predict한 class name)
		class_prediction = pred_P  
		class_prediction = tf.argmax(class_prediction, axis=2)

		# 각 cell에 위치한 각 Bbox의 confidence score 계산
		# confidence_score = predicted object confidence * predited class probability
		confidence_score = np.zeros_like(pred_C)
		for i in range(boxes_per_cell):  
			confidence_score[:, :, i] = (pred_C[:, :, i] * pred_P[:, :, i])

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
					pred_confidence_score = confidence_score[i][j][k]
                    
					# for문이 끝나면 bounding_box_info_list에는 7(cell_size) * 7(cell_size) * 2) = 98 개의 bounding box의 information이 들어있다.
					# 각 bounding box의 information은 (x, y, w, h, class_name, confidence)이다.
					# add bounding box dict list
					bounding_box_info_list.append(yolo_format_to_bounding_box_dict(pred_xcenter, 
																				   pred_ycenter,
																				   pred_box_w,
																				   pred_box_h,
																				   pred_class_name,
																				   pred_confidence_score))

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
			# 특정 class에만 ground truth bounding box information을 draw
			for label_num in class_name_dict.keys():
				if int(class_label) == label_num:     
					ground_truth_bounding_box_info_list.append(
						yolo_format_to_bounding_box_dict(xcenter, ycenter, box_w, box_h,
						 str(class_name_dict[label_num]), 1.0))

		ground_truth_drawing_image = drawing_image.copy()
		# draw ground-truth image
		# window에 정답값의 bounding box와 그에 따른 information을 draw
		for ground_truth_bounding_box_info in ground_truth_bounding_box_info_list:
			draw_bounding_box_and_label_info(
				ground_truth_drawing_image,
				ground_truth_bounding_box_info['left'],
				ground_truth_bounding_box_info['top'],
				ground_truth_bounding_box_info['right'],
				ground_truth_bounding_box_info['bottom'],
				ground_truth_bounding_box_info['class_name'],
				ground_truth_bounding_box_info['confidence_score'],
				color_list[cat_class_to_label_dict[ground_truth_bounding_box_info['class_name']]])
		 
		# find one max confidence bounding box
		# Non-maximum suppression을 사용하지 않고, 약식으로 진행 (confidence 상위 두 개의 bounding box 선택)
		confidence_bounding_box_list = find_enough_confidence_bounding_box(bounding_box_info_list,
																		   FLAGS.tensorboard_log_path,
																		   ckpt.step,
																		   validation_image_index)


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
		# 두 이미지를 연결(왼쪽엔 ground_truth, 오른쪽엔 drawing_image)
		drawing_image = drawing_image / 255
		drawing_image = tf.expand_dims(drawing_image, axis = 0)

		# save tensorboard log
		with validation_summary_writer.as_default():
			tf.summary.image('validation_image_'+str(validation_image_index), drawing_image, step=int(ckpt.step))
		
		detection_num, class_num, detection_rate = performance_evaluation(confidence_bounding_box_list,
																		  object_num,
																		  labels,
																		  class_name_to_label_dict,
																		  validation_image_index)
		success_detection_num += detection_num
		correct_answers_class_num += class_num
		detection_rate_sum +=detection_rate

	average_detection_rate = detection_rate / num_visualize_image  				# 평균 object detection 비율	
	perfect_detection_accuracy = success_detection_num / num_visualize_image	# 완벽한 object detection이 이루어진 비율
	classification_accuracy = correct_answers_class_num / num_visualize_image 	# 정확한 classicifiation이 이루어진 비율
	
	with average_detection_rate_writer.as_default():
		print(f"average_detection_rate: {average_detection_rate}")
		tf.summary.scalar('average_detection_rate', average_detection_rate, step=int(ckpt.step))

	with perfect_detection_accuracy_writer.as_default():
		print(f"perfect_detection_accuracy: {perfect_detection_accuracy}")
		tf.summary.scalar('perfect_detection_accuracy', perfect_detection_accuracy, step=int(ckpt.step))

	with classification_accuracy_writer.as_default():
		print(f"classification_accuracy: {classification_accuracy}")
		tf.summary.scalar('classification_accuracy', classification_accuracy, step=int(ckpt.step))

    
    
def main(_): 
	# set learning rate decay
	lr_schedule = schedules.ExponentialDecay(
		FLAGS.init_learning_rate,
		decay_steps=FLAGS.lr_decay_steps,
		decay_rate=FLAGS.lr_decay_rate,
		staircase=True)

	# set optimizer
	optimizer = Adam(lr_schedule) 
	class_loss_object = CategoricalCrossentropy()
	confidence_loss_object = BinaryCrossentropy()

    # set directory path
	(checkpoint_path,
	 train_summary_writer,
	 validation_summary_writer,
	 average_detection_rate_writer,
	 perfect_detection_accuracy_writer,
	 classification_accuracy_writer) = dir_setting(dir_name, 
											 CONTINUE_LEARNING,
											 FLAGS.checkpoint_path, 
											 FLAGS.tensorboard_log_path)

	ckpt, ckpt_manager, YOLOv1_model = set_checkpoint_manager(input_height,
						   									  input_width,
															  cell_size,
						   									  boxes_per_cell,
															  num_classes,
															  checkpoint_path)

	for epoch in range(FLAGS.num_epochs):
		num_batch = len(list(train_data))
		for iter, features in enumerate(train_data):
			batch_image = features['image']
			batch_bbox = features['objects']['bbox']
			batch_labels = features['objects']['label']

			batch_image = tf.squeeze(batch_image, axis=1)
			batch_bbox = tf.squeeze(batch_bbox, axis=1)
			batch_labels = tf.squeeze(batch_labels, axis=1)

			batch_labels = remove_irrelevant_label(batch_labels, class_name_dict)

			# run optimization and compute loss
			(total_loss, 
			 coord_loss, 
			 object_loss, 
			 noobject_loss, 
			 class_loss) = train_step(optimizer, YOLOv1_model,
					   batch_image, batch_bbox, batch_labels,
					   class_loss_object,
					   confidence_loss_object)

			# print log
			print(f"Epoch: {epoch+1}, Iter: {(iter+1)}/{num_batch}, Loss: {total_loss.numpy()}")

			# save tensorboard log
			save_tensorboard_log(train_summary_writer, optimizer, ckpt,
								 total_loss, coord_loss, object_loss, noobject_loss, class_loss)

			# save checkpoint
			save_checkpoint(ckpt,ckpt_manager, FLAGS.save_checkpoint_steps)

			ckpt.step.assign_add(1) # epoch나 train data의 개수와는 별개로, step 증가
			
            # occasionally check validation data and save tensorboard log
			if iter % FLAGS.validation_steps == 0:
				save_validation_result(YOLOv1_model,
									   ckpt, 
									   validation_summary_writer,
									   average_detection_rate_writer,
									   perfect_detection_accuracy_writer,
									   classification_accuracy_writer,
									   FLAGS.num_visualize_image,
									   class_loss_object,
									   confidence_loss_object
									   )


if __name__ == '__main__':  
	app.run(main) # main함수 실행
