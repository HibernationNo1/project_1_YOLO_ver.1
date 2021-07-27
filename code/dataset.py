import tensorflow as tf
import numpy as np

import tensorflow_datasets as tfds

# dict of classes to detect 
class_name_dict = {
	7: "cat", 9:"cow"
}

def predicate(x):  # x는 하나의 data.
	label = x['objects']['label']
	
	# 7또는 9라는 label의 object가 하나라도 포함 된 data는 모두 추려낸다.	
	reduced_sum = 0.0

	for label_num in class_name_dict.keys():
		isallowed = tf.equal(tf.constant([float(label_num)]), tf.cast(label, tf.float32)) # label이 label_num인 element만 True
		reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32)) 	# label이 class_num인 element의 개수
		reduced_sum += reduced

	return tf.greater(reduced_sum, tf.constant(0.))  # label이 7인 element의 개수가 0보다 클 때(1개 이상일때) True


# load pascal voc2007/voc2012 dataset using tfds

def load_pascal_voc_dataset(batch_size):  
    # set dataset for training
	voc2007_test_split_data = tfds.load("voc/2007", split=tfds.Split.TEST, batch_size=1)
	voc2012_train_split_data = tfds.load("voc/2012", split=tfds.Split.TRAIN, batch_size=1)
	voc2012_validation_split_data = tfds.load("voc/2012", split=tfds.Split.VALIDATION, batch_size=1)

	train_data = voc2007_test_split_data.concatenate(voc2012_train_split_data).concatenate(voc2012_validation_split_data)
    
    # set data for validation
	voc2007_validation_split_data = tfds.load("voc/2007", split=tfds.Split.VALIDATION, batch_size=1)
	validation_data = voc2007_validation_split_data
    
	train_data = train_data.filter(predicate) # 7 label 만 정제해서 할당 
	train_data = train_data.padded_batch(batch_size) # train_data가 filter(predicate) 에 의해 가변적인 크기를 가지고 있으므로 batch 대신 padded_batch 사용

	validation_data = validation_data.filter(predicate)
	validation_data = validation_data.padded_batch(batch_size)
    
	return train_data, validation_data


def load_pascal_voc_dataset_for_test(batch_size):
	voc2007_train_split_data = tfds.load("voc/2007", split=tfds.Split.TRAIN, batch_size=1)
	test_data = voc2007_train_split_data

	test_data = test_data.filter(predicate)
	test_data = test_data.padded_batch(batch_size)
	return test_data


def bounds_per_dimension(ndarray):
	return map(
    	lambda e: range(e.min(), e.max() + 1),
    	np.where(ndarray != 0)
  	)


def zero_trim_ndarray(ndarray):
	return ndarray[np.ix_(*bounds_per_dimension(ndarray))]

def index_reorder(labels):
	tmp = np.zeros_like(labels)
	num = 0
	for i in range(tf.shape(labels)[0]):
		if not labels[i] == 0:
			tmp[num] = labels[i]
			num +=1
	labels = tf.constant(tmp)
	return labels

def process_each_ground_truth(original_image, 
                              bbox,
                              class_labels,
                              input_width,   
                              input_height
                              ):
    
	# image에 zero padding 제거
	image = original_image.numpy()
	image = zero_trim_ndarray(image)

	# set original width height
	original_w = image.shape[1] # original image의 width
	original_h = image.shape[0] # original image의 height

	# image의 x, y center coordinate를 계산하기 위해 rate compute
	width_rate = input_width * 1.0 / original_w 
	height_rate = input_height * 1.0 / original_h

	# YOLO input size로 image resizing
	image = tf.image.resize(image, [input_height, input_width])

	# object_num = np.count_nonzero(bbox, axis=0)[0]
	object_num = np.count_nonzero(class_labels, axis=0)

	# class_num = 2 일 때 tf.shape(class_labels) = (6,) , tf.shape(Bbox) = (6,4) 임을 고려
	# [0 7 0 0 0 0] 을 [7 0 0 0 0 0] 처럼 index를 재정렬하는 function
	class_labels = index_reorder(class_labels)

	tmp = np.zeros_like(bbox)
	for i in range(tf.shape(bbox)[1]):
		tmp[:, i] = index_reorder(bbox[:, i])
	bbox = tf.constant(tmp)

	# labels initialize
	labels = [[0, 0, 0, 0, 0]] * object_num # (x, y, w, h, class_number) * object_num 
    
	for i in range(object_num):
		# 0~1 사이로 표현되어있던 각 coordinate를 pixel단위의 coordinate로 표현
		xmin = bbox[i][1] * original_w
		ymin = bbox[i][0] * original_h
		xmax = bbox[i][3] * original_w
		ymax = bbox[i][2] * original_h

		class_num = class_labels[i] # 실제 class labels

		# ont_hot encoding
		num_of_class = len(class_name_dict.keys()) 
		index_list = [n for n in range(num_of_class)]
		oh_class_num = (tf.one_hot(tf.cast((index_list), tf.int32), num_of_class, dtype=tf.float32))
		for j in range(num_of_class): 
			if int(class_num) == list(class_name_dict.keys())[j]:
				class_num = oh_class_num[j]
				break

		# resizing 된 image에 맞는 center coordinate 
		xcenter = (xmin + xmax) * 1.0 / 2 * width_rate 
		ycenter = (ymin + ymax) * 1.0 / 2 * height_rate

		# resizing 된 image에 맞는 bounding box의 width, height 
		box_w = (xmax - xmin) * width_rate
		box_h = (ymax - ymin) * height_rate

		# YOLO format형태의 5가지 labels 완성
		labels[i] = [xcenter, ycenter, box_w, box_h, class_num]

	return [image.numpy(), labels, object_num]
