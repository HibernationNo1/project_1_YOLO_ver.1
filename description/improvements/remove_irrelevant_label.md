**labels에서 target class외의 모든 label을 loss계산에서 제외하여 학습 효율 개선**



dataset.py의 `load_pascal_voc_dataset`에서 사용한 `filter`의 `predicate` function의 동작은 target class object가 하나라도 포함 된 data는 모두 추려낸다. 이 과정에서 target class가 아닌 class의 object도 포함된 data가 있는데, 이러한 object는 학습 목표과 관련이 없는 label이기 때문에 loss를 높히는 원인이 된다고 생각했다. 



### improvement

해결 방법으로, class_labels안의 value중, target class외의 모든 element를 0으로 만드는 function을 정의했다.



ex)

> detection target class가 7(cat), 9(cow일 때)
>
> `train_data`의 `feature[object][label]`을 확인해보면 4, 14, 와 같이 7, 9외의 element가 있는 것을 확인할 수 있다.
>
> class_labels == [11 7 14 9 0 0]
>
> 일 때 class_labels == [0 7 0 9 0 0] 으로 만들어준다.



##### **remove_irrelevant_label**

detect할 class에 대한 label만 추려내고, 나머지 label은 0으로 만드는 function.

Bbox label에도 target label이외의 element는 0으로 만들었다.

```python
def remove_irrelevant_label(batch_bbox, batch_labels, class_name_dict):
	tmp_labels = np.zeros_like(batch_labels)
	tmp_bbox = np.zeros_like(batch_bbox)

	for i in range(int(tf.shape(batch_labels)[0])): 		# image 1개당
		for j in range(int(tf.shape(batch_labels)[1])):		# object 1개당
			for lable_num in class_name_dict.keys(): 
				if batch_labels[i][j] == lable_num:
					tmp_labels[i][j] = batch_labels[i][j]
					tmp_bbox[i][j] = batch_bbox[i][j]
					continue

	batch_labels = tf.constant(tmp_labels)
	batch_bbox = tf.constant(tmp_bbox)

	return batch_bbox, batch_labels
```



또한 `dataset.py`의 `process_each_ground_truth`에서  

`object_num = np.count_nonzero(bbox, axis=0)[0]`을`object_num = np.count_nonzero(class_labels, axis=0)`으로 변경해야 한다.



`object_num`의 기준이 bbox라면, bbox 안에는 train data에 학습 목표과 관련이 없는 label이 포함되어 있지만, 

`object_num`의 기준을 class_labels로 결정하면 자연스럽게 irrelevant한 object에 대한 coordnate와 width, height도 추가되지 않기 때문에 loss가 줄어들게 된다.



##### index_reorder

`class_labels == [0 7 0 0 0 0]`  또는  `class_labels == [0 7 0 9 0 0]` 와 같이 앞의 index에 위치한 element가 0이 있는 `class_labels` 일 경우, labels initialize 과정에서 for문의 iteration 횟수가 부족해 labeling이 제대로 되지 않는 경우가 발생할 수 있다.

이런 경우 

`class_labels == [0 7 0 0 0 0]` 는 `class_labels == [7 0 0 0 0 0]` 으로, 

`class_labels == [0 7 0 9 0 0]` 는 `class_labels == [7 9 0 0 0 0]` 

으로 변경해주어야 한다. (Bbox의 index도 마찬가지)



1. dataset.py에서 index를 재정렬해주는 function `index_reorder` 을 define

   ```python
   def index_reorder(labels):
   	tmp = np.zeros_like(labels)
   	num = 0
   	for i in range(tf.shape(labels)[0]):
   		if not labels[i] == 0:
   			tmp[num] = labels[i]
   			num +=1
   	labels = tf.constant(tmp)
   	return labels
   ```

2. `process_each_ground_truth` 안에서

   `object_num = np.count_nonzero(class_labels, axis=0)` 다음에 

   `index_reorder`으로 `class_labels`과 `Bbox`의 index를 재정렬

   ```python
   	object_num = np.count_nonzero(class_labels, axis=0)
   
   	# class_num = 2 일 때 tf.shape(class_labels) = (6,) , tf.shape(Bbox) = (6,4) 임을 고려
   	# [0 7 0 0 0 0] 을 [7 0 0 0 0 0] 처럼 index를 재정렬하는 function
   	class_labels = index_reorder(class_labels)
   
   	tmp = np.zeros_like(bbox)
   	for i in range(tf.shape(bbox)[1]):
   		tmp[:, i] = index_reorder(bbox[:, i])
   	bbox = tf.constant(tmp)
   ```

   

   ex)

   ```
   Bbox 재정렬 전
   tf.Tensor(
   [[0.   0.   0.   0.  ]
    [0.   0.   0.   0.  ]
    [0.3  0.4  0.9  0.78]
    [0.   0.   0.   0.  ]
    [0.1  0.2  0.5  0.6 ]
    [0.   0.   0.   0.  ]], shape=(6, 4), dtype=float32)
    
   Bbox 재정렬 후
   tf.Tensor(
   [[0.3  0.4  0.9  0.78]
    [0.1  0.2  0.5  0.6 ]
    [0.   0.   0.   0.  ]
    [0.   0.   0.   0.  ]
    [0.   0.   0.   0.  ]
    [0.   0.   0.   0.  ]], shape=(6, 4), dtype=float32)
   ```

   

##### label number check code

```python
import tensorflow as tf
import cv2
import tensorflow_datasets as tfds

def predicate(x):  # x는 하나의 data.
	label = x['objects']['label']
	
	# 7또는 9라는 label의 object가 하나라도 포함 된 data는 모두 추려낸다.	
	reduced_sum = 0.0


	isallowed = tf.equal((tf.constant([float(13)])), tf.cast(label, tf.float32)) # label이 label_num인 element만 True
	reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32)) 	# label이 class_num인 element의 개수
	reduced_sum += reduced

	return tf.greater(reduced_sum, tf.constant(0.))  # label이 7인 element의 개수가 0보다 클 때(1개 이상일때) True


voc2007_test_split_data = tfds.load("voc/2007", split=tfds.Split.TEST, batch_size=1)
voc2012_train_split_data = tfds.load("voc/2012", split=tfds.Split.TRAIN, batch_size=1)
voc2012_validation_split_data = tfds.load("voc/2012", split=tfds.Split.VALIDATION, batch_size=1)

train_data = voc2007_test_split_data.concatenate(voc2012_train_split_data).concatenate(voc2012_validation_split_data)

train_data = train_data.filter(predicate) # 7 label 만 정제해서 할당 
train_data = train_data.padded_batch(12)

for iter, features in enumerate(train_data):
	batch_image = features['image']
	batch_image = tf.squeeze(batch_image, axis=1)[iter, :, :, :].numpy()
	
	cv2.imshow('image', batch_image)
	while True:
		if cv2.waitKey() == 27:  # 27은 esc의 아스키 코드임
			break

# 7: "cat"
# 11: "dog"
# 12: "horse"
# 13: "bike"
# 14: "human"
```

