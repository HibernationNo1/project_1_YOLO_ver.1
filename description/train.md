# train.py

training을 진행하며, 반복이 특정 횟수에 도달하면 validation을 수행한다.

모든 training log와 validation log, label과 prediction image간의 비교를 tensorboard에 저장한다.



**import**

```python
import tensorflow as tf
import os
import numpy as np
import random

from absl import flags
from absl import app

from tensorflow.keras.optimizers import schedules
from tensorflow.keras.optimizers import Adam


from utils import generate_color, dir_setting, save_checkpoint
from dataset import load_pascal_voc_dataset
from loss import yolo_loss
from dataset import process_each_ground_truth
from utils import (draw_bounding_box_and_label_info,
				   find_max_confidence_bounding_box, 
				   yolo_format_to_bounding_box_dict)
```



**define function**

- [calculate_loss](#calculate_loss)
- [train_step](#train_step)
- [save_tensorboard_log](#save_tensorboard_log)
- [save_validation_result](#save_validation_result)
- [main(_)](#main(_))
- [Full code](#'Full code')



## set parameters

training에 필요한 parameter를 정의하고, parameter를 담을 dictionary를 초기화한다.



```python
#flags instance로 hyper parameters setting
flags.DEFINE_string('checkpoint_path', default='saved_model', help='path to a directory to save model checkpoints during training')
flags.DEFINE_integer('save_checkpoint_steps', default=50, help='period at which checkpoints are saved (defaults to every 50 steps)')
flags.DEFINE_string('tensorboard_log_path', default='tensorboard_log', help='path to a directory to save tensorboard log')
flags.DEFINE_integer('validation_steps', default=50, help='period at which test prediction result and save image')  # 몇 번의 step마다 validation data로 test를 할지 결정
flags.DEFINE_integer('num_epochs', default=30, help='training epochs') # original paper : 135 epoch
flags.DEFINE_float('init_learning_rate', default=0.0001, help='initial learning rate') # original paper : 0.001 (1epoch) -> 0.01 (75epoch) -> 0.001 (30epoch) -> 0.0001 (30epoch)
flags.DEFINE_float('lr_decay_rate', default=0.5, help='decay rate for the learning rate')
flags.DEFINE_integer('lr_decay_steps', default=2000, help='number of steps after which the learning rate is decayed by decay rate') # 2000번 마다 init_learning_rate * lr_decay_rate 을 실행
# 2000 step : init_learning_rate = 0.00005, 4000 step : init_learning_rate = 0.000025
flags.DEFINE_integer('num_visualize_image', default=8, help='number of visualize image for validation')
# 중간중간 validation을 할 때마다 몇 개의 batch size로 visualization을 할지 결정하는 변수

FLAGS = flags.FLAGS


# set cat label dictionary 
cat_label_dict = {
  0: "cat"
}
cat_class_to_label_dict = {v: k for k, v in cat_label_dict.items()}

dir_name = 'tmp'
CONTINUE_LEARNING = False  # 이전에 했던 training을 다시 시작할 때 False, 계속 이어서 할 땐 True 

# set configuration value
batch_size = 24 	# original paper : 64
input_width = 224 	# original paper : 448
input_height = 224 	# original paper : 448
cell_size = 7
num_classes = 1 	# original paper : 20
boxes_per_cell = 2

# set color_list for drawing
color_list = generate_color(num_classes)

# generate dataset
train_data, validation_data = load_pascal_voc_dataset(batch_size)

# set loss function coefficients
coord_scale = 10 	# original paper : 5  
class_scale = 0.1  	# original paper : 1
object_scale = 1	# original paper : None
noobject_scale = 0.5	# original paper : None

loss_dict = dict()  # loss를 보관할 dict
loss_dict['total_loss'] = 0.0
loss_dict['coord_loss'] = 0.0
loss_dict['object_loss'] = 0.0 
loss_dict['noobject_loss'] = 0.0 
loss_dict['class_loss'] = 0.0 


if __name__ == '__main__':  
	app.run(main) # main함수 실행
```



**detail**

- line 18 : object detection에서 자주 사용되는 dict 패턴으로, computer가 인지한 숫자를 사람이 알아볼 수 있게 `key`는 integer,` value`는 string 으로 setting한다.

- line 21 : 위의 `cat_label_dict`의 `key`와 `value`의 위치(역할)을 바꾼 dictionary로, 해당 code에서는 cat에 대한 class만 classification할 것이기 때문에 cat만 setting한다.

- line 54 : 해당 code가 쓰여진 file이 entry point이기 때문에 main함수 실행

  **실제로 해당 if문은 train.py의 마지막에 사용되어야 한다.**



## calculate_loss

data를 parsing한 후 loss값을 계산하여 dictionary에 저장하는 function

`loss_dict` 가 전역에 선언되어 있으니 return값이 없도록 구현했다.



**input argument description**

- `model` : instance of model class
- `batch_image` : train data 
- `batch_bbox` : batch bounding box
- `batch_labels` : label data



```python
def calculate_loss(model, batch_image, batch_bbox, batch_labels):
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
			 each_object_class_loss) = yolo_loss(predict[0],
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
								   				 class_scale )
			
            # 각각 전체의 batch에 대해서 loss 합산
			loss_dict['total_loss'] = loss_dict['total_loss'] + each_object_total_loss
			loss_dict['coord_loss'] = loss_dict['coord_loss'] + each_object_coord_loss
			loss_dict['object_loss'] = loss_dict['object_loss'] + each_object_object_loss
			loss_dict['noobject_loss'] = loss_dict['noobject_loss'] + each_object_noobject_loss
			loss_dict['class_loss'] = loss_dict['class_loss'] + each_object_class_loss
```



**detail**

- line 3 :

  - `process_each_ground_truth` : 원하는 data를 parsing하는 function
  - `image` : resize된 data
  - `labels` : 절대 좌표로 표현된 label값
  - `object_num` : object 개수

- line 8 : `expand_dims`을 사용해서 0차원에 dummy dimension 추가하는 이유

  `image`는 결국  `tf.keras.applications.InceptionV3` 의 input으로 들어가서 처리되는데, 이때 인풋으로 기대하는 dimension이 `[batch_size, height, width, color_channel]`이라 앞에 dummy로 `batch_size=1`을 넣어줘서 차원을 맞춰주어 오류가 나지 않도록 하기 위함이다.

- line 12 : predict의 shape = `[1, cell_size, cell_size, num_classes, 5 * boxes_per_cell]`

  `tf.shape(predict)` : tf.Tensor([ 1  7  7 11], shape=(4,), dtype=int32)

  `tf.shape(predict[0])` : tf.Tensor([ 7  7 11], shape=(3,), dtype=int32)



## train_step

gradient descent을 수행하는 function



```python
def train_step(optimizer, model, batch_image, batch_bbox, batch_labels): 
	with tf.GradientTape() as tape:
		calculate_loss(model, batch_image, batch_bbox, batch_labels )
	
	gradients = tape.gradient(loss_dict['total_loss'], model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```



**detail**

- line 5 :

  ` model.trainable_variables` 을 통해 instance`YOLOv1_model` 의 weight와 bias에 접근하여 

  `total_loss` 에 대한 weight와 bias의 편미분 값을 계산한다.

- line 6 :

  line 5에서 계산된 `gradients` 을 통해 parameter 최적화를 1회 진행한다.



## save_tensorboard_log

현재 시점의 step의 각 loss값을 tensorboard에 기록하는 function



- `ckpt` : checkpoint instance

```python
def save_tensorboard_log(train_summary_writer, optimizer, ckpt):
	with train_summary_writer.as_default():
		tf.summary.scalar('learning_rate ', optimizer.lr(ckpt.step).numpy(), step=int(ckpt.step))
		tf.summary.scalar('total_loss',	loss_dict['total_loss'], step=int(ckpt.step))
		tf.summary.scalar('coord_loss', loss_dict['coord_loss'], step=int(ckpt.step))
		tf.summary.scalar('object_loss ', loss_dict['object_loss'], step=int(ckpt.step))
		tf.summary.scalar('noobject_loss ', loss_dict['noobject_loss'], step=int(ckpt.step))
		tf.summary.scalar('class_loss ', loss_dict['class_loss'], step=int(ckpt.step)) 
```



## save_validation_result

- validation을 진행하고 그 결과를 저장

- ground-truth image와 prediction한 image를 좌, 우 배치하여 비교하는 사진을 저장





**input argument description**

- `model` : `YOLOv1_model` 의 instance

- `ckpt` : checkpoint instance

- `validation_summary_writer` : tensorboard 에 log를 남기기 위한 writer instance

- `num_visualize_image` : `flags` 로 선언된, validation을 할 때마다 몇 개의 batch size로 visualization을 할지 결정하는 변수

  > 해당 코드에서는 8로 정의되어있다.

```python
def save_validation_result(model, ckpt, validation_summary_writer, num_visualize_image):
	total_validation_total_loss = 0.0
	total_validation_coord_loss = 0.0  
	total_validation_object_loss = 0.0
	total_validation_noobject_loss = 0.0  
	total_validation_class_loss = 0.0
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
		 validation_class_loss) = calculate_loss(model,
												 batch_validation_image,
												 batch_validation_bbox,
												 batch_validation_labels)
	   
		total_validation_total_loss = total_validation_total_loss + validation_total_loss
		total_validation_coord_loss = total_validation_coord_loss + validation_coord_loss
		total_validation_object_loss = total_validation_object_loss + validation_object_loss
		total_validation_noobject_loss = total_validation_noobject_loss + validation_noobject_loss
		total_validation_class_loss = total_validation_class_loss + validation_class_loss
	  
	# save validation tensorboard log
	with validation_summary_writer.as_default():
		tf.summary.scalar('total_validation_total_loss', total_validation_total_loss, step=int(ckpt.step))
		tf.summary.scalar('total_validation_coord_loss', total_validation_coord_loss, step=int(ckpt.step))
		tf.summary.scalar('total_validation_object_loss ', total_validation_object_loss, step=int(ckpt.step))
		tf.summary.scalar('total_validation_noobject_loss ', total_validation_noobject_loss, step=int(ckpt.step))
		tf.summary.scalar('total_validation_class_loss ', total_validation_class_loss, step=int(ckpt.step))
	  
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
		predict = tf.reshape(predict,
				 [tf.shape(predict)[0], cell_size, cell_size, num_classes + 5 * boxes_per_cell])
	
		# parse prediction
		predict_boxes = predict[0, :, :, num_classes + boxes_per_cell:]
		predict_boxes = tf.reshape(predict_boxes, [cell_size, cell_size, boxes_per_cell, 4])
	
		confidence_boxes = predict[0, :, :, num_classes:num_classes + boxes_per_cell]
		confidence_boxes = tf.reshape(confidence_boxes, [cell_size, cell_size, boxes_per_cell, 1])
	   
		# Non-maximum suppression
		class_prediction = predict[0, :, :, 0:num_classes]
		class_prediction = tf.argmax(class_prediction, axis=2)
		  
		# make prediction bounding box list
		bounding_box_info_list = []
		for i in range(cell_size):
			for j in range(cell_size):
				for k in range(boxes_per_cell):
					pred_xcenter = predict_boxes[i][j][k][0]
					pred_ycenter = predict_boxes[i][j][k][1]
					pred_box_w = tf.minimum(input_width * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][2]))
					pred_box_h = tf.minimum(input_height * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][3]))
				   
					
					pred_class_name = cat_label_dict[class_prediction[i][j].numpy()]                   
					pred_confidence = confidence_boxes[i][j][k].numpy()[0]
                    
					# for문이 끝나면 bounding_box_info_list에는 7(cell_size) * 7(cell_size) * 2(boxes_per_cell) = 98 개의 bounding box의 information이 들어있다.
					# add bounding box dict list
					bounding_box_info_list.append(yolo_format_to_bounding_box_dict(pred_xcenter, 
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
			if class_label == 7:     # label 7 : cat
				ground_truth_bounding_box_info_list.append(
					yolo_format_to_bounding_box_dict(xcenter, ycenter, box_w, box_h, 'cat', 1.0))
		
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
				ground_truth_bounding_box_info['confidence'],
				color_list[cat_class_to_label_dict[ground_truth_bounding_box_info['class_name']]])
		 
		# find one max confidence bounding box
		# Non-maximum suppression을 사용하지 않고, 약식으로 진행 (confidence가 가장 큰 bounding box 하나만 선택)
		max_confidence_bounding_box = find_max_confidence_bounding_box(bounding_box_info_list)

		# draw prediction (image 위에 bounding box 표현)
		draw_bounding_box_and_label_info(
			drawing_image,
			max_confidence_bounding_box['left'],
			max_confidence_bounding_box['top'],
			max_confidence_bounding_box['right'],
			max_confidence_bounding_box['bottom'],
			max_confidence_bounding_box['class_name'],
			max_confidence_bounding_box['confidence'],
			color_list[cat_class_to_label_dict[max_confidence_bounding_box['class_name']]])
	 

		# left : ground-truth, right : prediction
		drawing_image = np.concatenate((ground_truth_drawing_image, drawing_image), axis=1)
		# 두 이미지를 연결(왼쪽엔 ground_truth, 오른쪽엔 drawing_image)
		drawing_image = drawing_image / 255
		drawing_image = tf.expand_dims(drawing_image, axis = 0)

		# save tensorboard log
		with validation_summary_writer.as_default():
			tf.summary.image('validation_image_'+str(validation_image_index), drawing_image, step=int(ckpt.step))
```





## main(_)

training을 진행하는 main function



```python
def main(_): 
	# set learning rate decay
	lr_schedule = schedules.ExponentialDecay(
		FLAGS.init_learning_rate,
		decay_steps=FLAGS.lr_decay_steps,
		decay_rate=FLAGS.lr_decay_rate,
		staircase=True)

	# set optimizer
	optimizer = Adam(lr_schedule) 

    # set directory path
	(checkpoint_path,
	train_summary_writer, 
	validation_summary_writer) = dir_setting(dir_name, 
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

			# run optimization and compute loss
			train_step(optimizer, YOLOv1_model,
					   batch_image, batch_bbox, batch_labels)

			# print log
			print(f"Epoch: {epoch+1}, Iter: {iter+1}, Loss: {loss_dict['total_loss'].numpy()}")

			# save tensorboard log
			save_tensorboard_log(train_summary_writer, optimizer, ckpt)

			# save checkpoint
			save_checkpoint(ckpt,ckpt_manager, FLAGS.save_checkpoint_steps)

			ckpt.step.assign_add(1) # epoch나 train data의 개수와는 별개로, step 증가
			
            # occasionally check validation data and save tensorboard log
			if iter % FLAGS.validation_steps == 0:
				save_validation_result(YOLOv1_model, ckpt, validation_summary_writer, FLAGS.num_visualize_image)

```



**detail**

- line 3 : learning rate의 detail을 결정

  `initial learning rate` = 0.0001 이며, 2000번의 step마다 0.5씩 곱한다.

  `default steps` = 2000, `decay_rate` = 0.5

- line 10 : original paper에서는 ` optimizer = tf.optimizers.SGD(lr = 0.01, momentum = 0.9, decay = 0.0005)`

- line 53 : 반복이 validation_steps에 도달하면, 현재 step의 기준으로 model의 parameter에 기반한 validation을 진행





## Full code



```python
import tensorflow as tf
import os
import numpy as np
import random

from absl import flags
from absl import app

from tensorflow.keras.optimizers import schedules
from tensorflow.keras.optimizers import Adam


from utils import generate_color, dir_setting, save_checkpoint
from dataset import load_pascal_voc_dataset
from loss import yolo_loss
from dataset import process_each_ground_truth
from utils import (draw_bounding_box_and_label_info,
				   find_max_confidence_bounding_box, 
				   yolo_format_to_bounding_box_dict)

#flags instance로 hyper parameters setting
flags.DEFINE_string('checkpoint_path', default='saved_model', help='path to a directory to save model checkpoints during training')
flags.DEFINE_integer('save_checkpoint_steps', default=50, help='period at which checkpoints are saved (defaults to every 50 steps)')
flags.DEFINE_string('tensorboard_log_path', default='tensorboard_log', help='path to a directory to save tensorboard log')
flags.DEFINE_integer('validation_steps', default=50, help='period at which test prediction result and save image')  # 몇 번의 step마다 validation data로 test를 할지 결정
flags.DEFINE_integer('num_epochs', default=30, help='training epochs') # original paper : 135 epoch
flags.DEFINE_float('init_learning_rate', default=0.0001, help='initial learning rate') # original paper : 0.001 (1epoch) -> 0.01 (75epoch) -> 0.001 (30epoch) -> 0.0001 (30epoch)
flags.DEFINE_float('lr_decay_rate', default=0.5, help='decay rate for the learning rate')
flags.DEFINE_integer('lr_decay_steps', default=2000, help='number of steps after which the learning rate is decayed by decay rate') # 2000번 마다 init_learning_rate * lr_decay_rate 을 실행
# 2000 step : init_learning_rate = 0.00005, 4000 step : init_learning_rate = 0.000025
flags.DEFINE_integer('num_visualize_image', default=8, help='number of visualize image for validation')
# 중간중간 validation을 할 때마다 몇 개의 batch size로 visualization을 할지 결정하는 변수

FLAGS = flags.FLAGS


# set cat label dictionary 
cat_label_dict = {
  0: "cat"
}
cat_class_to_label_dict = {v: k for k, v in cat_label_dict.items()}

dir_name = 'train1'
CONTINUE_LEARNING = False  # 이전에 했던 training을 다시 시작할 때 False, 계속 이어서 할 땐 True 

# set configuration value
batch_size = 24 	# original paper : 64
input_width = 224 	# original paper : 448
input_height = 224 	# original paper : 448
cell_size = 7
num_classes = 1 	# original paper : 20
boxes_per_cell = 2

# set color_list for drawing
color_list = generate_color(num_classes)

# generate dataset
train_data, validation_data = load_pascal_voc_dataset(batch_size)

# set loss function coefficients
coord_scale = 10 	# original paper : 5  
class_scale = 0.1  	# original paper : 1
object_scale = 1	# original paper : None
noobject_scale = 0.5	# original paper : None

loss_dict = dict()  # loss를 보관할 dict
loss_dict['total_loss'] = 0.0
loss_dict['coord_loss'] = 0.0
loss_dict['object_loss'] = 0.0 
loss_dict['noobject_loss'] = 0.0 
loss_dict['class_loss'] = 0.0 


def train_step(optimizer, model, batch_image, batch_bbox, batch_labels): 
	with tf.GradientTape() as tape:
		calculate_loss(model, batch_image, batch_bbox, batch_labels )
	
	gradients = tape.gradient(loss_dict['total_loss'], model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    
def save_tensorboard_log(train_summary_writer, optimizer, ckpt):
	with train_summary_writer.as_default():
		tf.summary.scalar('learning_rate ', optimizer.lr(ckpt.step).numpy(), step=int(ckpt.step))
		tf.summary.scalar('total_loss',	loss_dict['total_loss'], step=int(ckpt.step))
		tf.summary.scalar('coord_loss', loss_dict['coord_loss'], step=int(ckpt.step))
		tf.summary.scalar('object_loss ', loss_dict['object_loss'], step=int(ckpt.step))
		tf.summary.scalar('noobject_loss ', loss_dict['noobject_loss'], step=int(ckpt.step))
		tf.summary.scalar('class_loss ', loss_dict['class_loss'], step=int(ckpt.step)) 
        
    
def save_validation_result(model, ckpt, validation_summary_writer, num_visualize_image):
	total_validation_total_loss = 0.0
	total_validation_coord_loss = 0.0  
	total_validation_object_loss = 0.0
	total_validation_noobject_loss = 0.0  
	total_validation_class_loss = 0.0
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
		 validation_class_loss) = calculate_loss(model,
												 batch_validation_image,
												 batch_validation_bbox,
												 batch_validation_labels)
	   
		total_validation_total_loss = total_validation_total_loss + validation_total_loss
		total_validation_coord_loss = total_validation_coord_loss + validation_coord_loss
		total_validation_object_loss = total_validation_object_loss + validation_object_loss
		total_validation_noobject_loss = total_validation_noobject_loss + validation_noobject_loss
		total_validation_class_loss = total_validation_class_loss + validation_class_loss
	  
	# save validation tensorboard log
	with validation_summary_writer.as_default():
		tf.summary.scalar('total_validation_total_loss', total_validation_total_loss, step=int(ckpt.step))
		tf.summary.scalar('total_validation_coord_loss', total_validation_coord_loss, step=int(ckpt.step))
		tf.summary.scalar('total_validation_object_loss ', total_validation_object_loss, step=int(ckpt.step))
		tf.summary.scalar('total_validation_noobject_loss ', total_validation_noobject_loss, step=int(ckpt.step))
		tf.summary.scalar('total_validation_class_loss ', total_validation_class_loss, step=int(ckpt.step))
	  
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
		predict = tf.reshape(predict,
				 [tf.shape(predict)[0], cell_size, cell_size, num_classes + 5 * boxes_per_cell])
	
		# parse prediction
		predict_boxes = predict[0, :, :, num_classes + boxes_per_cell:]
		predict_boxes = tf.reshape(predict_boxes, [cell_size, cell_size, boxes_per_cell, 4])
	
		confidence_boxes = predict[0, :, :, num_classes:num_classes + boxes_per_cell]
		confidence_boxes = tf.reshape(confidence_boxes, [cell_size, cell_size, boxes_per_cell, 1])
	   
		# Non-maximum suppression
		class_prediction = predict[0, :, :, 0:num_classes]
		class_prediction = tf.argmax(class_prediction, axis=2)
		  
		# make prediction bounding box list
		bounding_box_info_list = []
		for i in range(cell_size):
			for j in range(cell_size):
				for k in range(boxes_per_cell):
					pred_xcenter = predict_boxes[i][j][k][0]
					pred_ycenter = predict_boxes[i][j][k][1]
					pred_box_w = tf.minimum(input_width * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][2]))
					pred_box_h = tf.minimum(input_height * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][3]))
				   
					
					pred_class_name = cat_label_dict[class_prediction[i][j].numpy()]                   
					pred_confidence = confidence_boxes[i][j][k].numpy()[0]
                    
					# for문이 끝나면 bounding_box_info_list에는 7(cell_size) * 7(cell_size) * 2(boxes_per_cell) = 98 개의 bounding box의 information이 들어있다.
					# add bounding box dict list
					bounding_box_info_list.append(yolo_format_to_bounding_box_dict(pred_xcenter, 
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
			if class_label == 7:     # label 7 : cat
				ground_truth_bounding_box_info_list.append(
					yolo_format_to_bounding_box_dict(xcenter, ycenter, box_w, box_h, 'cat', 1.0))
		
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
				ground_truth_bounding_box_info['confidence'],
				color_list[cat_class_to_label_dict[ground_truth_bounding_box_info['class_name']]])
		 
		# find one max confidence bounding box
		# Non-maximum suppression을 사용하지 않고, 약식으로 진행 (confidence가 가장 큰 bounding box 하나만 선택)
		max_confidence_bounding_box = find_max_confidence_bounding_box(bounding_box_info_list)

		# draw prediction (image 위에 bounding box 표현)
		draw_bounding_box_and_label_info(
			drawing_image,
			max_confidence_bounding_box['left'],
			max_confidence_bounding_box['top'],
			max_confidence_bounding_box['right'],
			max_confidence_bounding_box['bottom'],
			max_confidence_bounding_box['class_name'],
			max_confidence_bounding_box['confidence'],
			color_list[cat_class_to_label_dict[max_confidence_bounding_box['class_name']]])
	 

		# left : ground-truth, right : prediction
		drawing_image = np.concatenate((ground_truth_drawing_image, drawing_image), axis=1)
		# 두 이미지를 연결(왼쪽엔 ground_truth, 오른쪽엔 drawing_image)
		drawing_image = drawing_image / 255
		drawing_image = tf.expand_dims(drawing_image, axis = 0)

		# save tensorboard log
		with validation_summary_writer.as_default():
			tf.summary.image('validation_image_'+str(validation_image_index), drawing_image, step=int(ckpt.step))
    
    
def main(_): 
	# set learning rate decay
	lr_schedule = schedules.ExponentialDecay(
		FLAGS.init_learning_rate,
		decay_steps=FLAGS.lr_decay_steps,
		decay_rate=FLAGS.lr_decay_rate,
		staircase=True)

	# set optimizer
	optimizer = Adam(lr_schedule) 

    # set directory path
	(checkpoint_path,
	train_summary_writer, 
	validation_summary_writer) = dir_setting(dir_name, 
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

			# run optimization and compute loss
			train_step(optimizer, YOLOv1_model,
					   batch_image, batch_bbox, batch_labels)

			# print log
			print(f"Epoch: {epoch+1}, Iter: {iter+1}, Loss: {loss_dict['total_loss'].numpy()}")

			# save tensorboard log
			save_tensorboard_log(train_summary_writer, optimizer, ckpt)

			# save checkpoint
			save_checkpoint(ckpt,ckpt_manager, FLAGS.save_checkpoint_steps)

			ckpt.step.assign_add(1) # epoch나 train data의 개수와는 별개로, step 증가
			
            # occasionally check validation data and save tensorboard log
			if iter % FLAGS.validation_steps == 0:
				save_validation_result(YOLOv1_model, ckpt, validation_summary_writer, FLAGS.num_visualize_image)
                
    
if __name__ == '__main__':  
	app.run(main) # main함수 실행
```


