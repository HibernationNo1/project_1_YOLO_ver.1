# dataset.py

using data set : **PASCAL VOC 2007, 2012**

Reference : http://host.robots.ox.ac.uk/pascal/VOC/voc2007/  ,  http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

**import**

```python
import tensorflow as tf
import numpy as np

import tensorflow_datasets as tfds
```



**define function**

- [load_pascal_voc_dataset](#load_pascal_voc_dataset)
- [predicate](#predicate)
- [process_each_ground_truth](#process_each_ground_truth)
- [zero_trim_ndarray](#zero_trim_ndarray)
- [bounds_per_dimension](#bounds_per_dimension)
- [Full code](#Full-code)



## load_pascal_voc_dataset

PASCAL VOC 2007, 2012 dataset을 load하고 preprocessing을 진행



### preprocessing

해당 project에서는 cat에 대한 calss만 분류를 진행할 것이기 때문에 **predicate** function을 통해 cat에 대한 data만 parsing을 진행한다.



**check number of each dataset**

```python
import tensorflow as tf

import tensorflow_datasets as tfds

_, ds_info2007 = tfds.load(name = "voc/2007", with_info = True)
_, ds_info2012 = tfds.load(name = "voc/2012", with_info = True)

print(ds_info2007.splits['train'].num_examples)		# 2501
print(ds_info2007.splits['test'].num_examples)		# 4952	
print(ds_info2007.splits['validation'].num_examples)# 2510

print(ds_info2012.splits['train'].num_examples) 	# 5717
print(ds_info2012.splits['test'].num_examples)		# 10991
print(ds_info2012.splits['validation'].num_examples)# 5823
```



#### train data

조금이라도 더 많은 data를 training에 사용하기 위해  train data 를 voc/2007_test_data + voc/2012_train_data + voc/2012_validation_data 로 구성했다.

- number of train data = 4952 + 5717 + 5823 



#### validation data

validation data로는 voc/2007_validation_data를 사용

- number of validation data = 2510



```python
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

```





## predicate

**load_pascal_voc_dataset**  에서 preprocessing을 진행할때 사용하기 위해 정의한 function이며, cat에 대한 data만 parsing하기 위해 allowed_labels에 7.0 의 value를 할당했다.

-  [voc/2007](https://pjreddie.com/media/files/VOC2007_doc.pdf) 의 paper을 보면 'cat' 은 8번째 class이며, index = 7 임을 알 수 있다.

```python
def predicate(x, allowed_labels=tf.constant([7.0])):
    label = x['objects']['label']
    isallowed = tf.equal(allowed_labels, tf.cast(label, tf.float32)) 	# label이 7인 element만 True

    reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32)) 			# label이 7인 element의 개수
	
    return tf.greater(reduced, tf.constant(0.))  # label이 7인 element의 개수가 1개 이상일 때 True
	
```



## process_each_ground_truth

YOLO model의 format형태 labels을 만들기 위한 function

- input argument description

  - `original_image` : train_data['image'] 를 의미한다.

  - `bbox` : features['objects']\['bbox']를 의미하며 

    **shape** : (max_object_num_in_batch, 4) == `[max_object_num_in_batch, [ymin / height, xmin / width, ymax / height, xmax / width]]`

    또한 각 coordinate는 image를 기준으로 normalize 된 상대좌표이다.

  - `class_labels` : one-hot encoding 되지 않은 class labels

  - `input_width` ,  `input_height` : width, height of image

- process result

  각 image 안의 각 object의 coordinate를 image기준의 상대좌표 →  image기준의 절대좌표로 변경 



```python
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

    object_num = np.count_nonzero(bbox, axis=0)[0]

    # labels initialize
    labels = [[0, 0, 0, 0, 0]] * object_num
    
    for i in range(object_num):
        xmin = bbox[i][1] * original_w
        ymin = bbox[i][0] * original_h
        xmax = bbox[i][3] * original_w
        ymax = bbox[i][2] * original_h

        class_num = class_labels[i] # 실제 class labels

		# set center coordinate 
        xcenter = (xmin + xmax) * 1.0 / 2 * width_rate 
        ycenter = (ymin + ymax) * 1.0 / 2 * height_rate

        # bounding box의 width, height 
        box_w = (xmax - xmin) * width_rate
        box_h = (ymax - ymin) * height_rate
		
        # YOLO format형태의 5가지 labels 완성
        labels[i] = [xcenter, ycenter, box_w, box_h, class_num]

    return [image.numpy(), labels, object_num]
```



**detail**

- line 18 :  image의 x, y center coordinate를 계산(line )하는 이유

  PASCAL VOC 의 labels coordinate는 image의 꼭지점 좌표로 표현되어 있다. 하지만 YOLO model은 bounding box의 중앙 coordinate로 표현하기 때문에 이를 위해 x, y의 center coordinate를 계산해야 한다.

   (input으로 들어가는 image의 width, height == 448 이다.)

- line 23  : `object_num` 은 전체 bounding box에서 object가 있는 box의 개수(즉, object 개수)이다. 

  zero element가 있는 이유 : batch_size로 받아오기 때문에, 어떤 image는 object가 3개이고, 다른 image에서는 object가 1개일 때 [1 1 1], [1 0 0] 이런 식으로 dimension을 맞추기 위해 zero가 있게 된다. 이를 빼주기 위해 count_nonzero 을 하는 것이다.

- line 26 : labels 의 shape == object 개수 * 5가지 element 

  element == (x, y, w, h, class num) 



## zero_trim_ndarray

**process_each_ground_truth**에서 

**bounds_per_dimension** 을 **ix_** function의 argument로 받아 zero padding을 제거한 후 다시 pixel단위로 재배열한다.

```python
def zero_trim_ndarray(ndarray):
	return ndarray[np.ix_(*bounds_per_dimension(ndarray))]
```





## bounds_per_dimension

받아온 ndarray(image)에서 0인 부분(zero padding)을 전부 제거하는 function



- zero padding이 있는 이유: batch_size로 받아오기 때문에, 작은 size의 image는 dims에 맞게 묶기기 위해서 (zero padding)으로 감싸져서 batch에 묶이게 된다.

```python
def bounds_per_dimension(ndarray):
	return map(
    	lambda e: range(e.min(), e.max() + 1),
    	np.where(ndarray != 0)
  	)
```

> np.where로 인해 0이 아닌 index의 list와, lambda함수가 mapping된다.





## Full code

```python
import tensorflow as tf
import numpy as np

import tensorflow_datasets as tfds

def predicate(x, allowed_labels=tf.constant([7.0])):
    label = x['objects']['label']
    isallowed = tf.equal(allowed_labels, tf.cast(label, tf.float32)) 	# label이 7인 element만 True

    reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32)) 			# label이 7인 element의 개수
	
    return tf.greater(reduced, tf.constant(0.))  # label이 7인 element의 개수가 1개 이상일 때 True

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


def bounds_per_dimension(ndarray):
	return map(
    	lambda e: range(e.min(), e.max() + 1),
    	np.where(ndarray != 0)
  	)


def zero_trim_ndarray(ndarray):
	return ndarray[np.ix_(*bounds_per_dimension(ndarray))]


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

    object_num = np.count_nonzero(bbox, axis=0)[0]

    # labels initialize
    labels = [[0, 0, 0, 0, 0]] * object_num
    
    for i in range(object_num):
        xmin = bbox[i][1] * original_w
        ymin = bbox[i][0] * original_h
        xmax = bbox[i][3] * original_w
        ymax = bbox[i][2] * original_h

        class_num = class_labels[i] # 실제 class labels

		# set center coordinate 
        xcenter = (xmin + xmax) * 1.0 / 2 * width_rate 
        ycenter = (ymin + ymax) * 1.0 / 2 * height_rate

        # bounding box의 width, height 
        box_w = (xmax - xmin) * width_rate
        box_h = (ymax - ymin) * height_rate
		
        # YOLO format형태의 5가지 labels 완성
        labels[i] = [xcenter, ycenter, box_w, box_h, class_num]

    return [image.numpy(), labels, object_num]
```

