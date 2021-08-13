# loss function.py

model의 parameter학습을 위한 loss function을 정의한다.

**total loss** = `coordinate loss` + `confidence loss` + `class loss`





##### Intersection over Union(IOU)

![img](https://t1.daumcdn.net/cfile/tistory/993477505D14A25016)

IoU = 교집합 영역 넓이 / 합집합 영역 넓이

best predicted bounding box란 모든 predicted bounding box 중에서 가장 큰 IOU 값을 가진 bounding box이다. 



**indicator function**

S*S size의 grid cell 중 특정 grid cell에서 믿을만한 bounding box만 살리고 나머진 버리는 용도로 사용된다.

i : i번째 grid cell (0부터 [cell]^2까지)

j : j 번째 detector (0부터 [Bboxs per cell] 까지)

- $$
  𝟙^{obj}_{ij}
  $$

  i 번째 grid cell에 object가 있고, 해당 cell 안에 j번째 detector가 있을 때에만 1을 return. 그 외에는 0

  > object가 있는 cell에서 j번째 detector가 있을 때에만 1

  ```python
  I = iou_predict_truth 	
  max_I = tf.reduce_max(I, 2, keepdims=True)
  best_box_mask = tf.cast((I >= max_I), tf.float32
                          
  object_exists_cell = np.zeros([cell_size, cell_size, 1])
  object_exists_cell_i, object_exists_cell_j = int(cell_size * ycenter / input_height), int(cell_size * xcenter / input_width)
  object_exists_cell[object_exists_cell_i][object_exists_cell_j] = 1
                          
  object_indicator_function = best_box_mask * object_exists_cell
  ```

  

- $$
  𝟙^{noobj}_{ij}
  $$

  i 번째 grid cell에 object가 없고, 해당 cell 안에 j번째 detector가 있을 때에만 1을 return. 그 외에는 0

  > object가 없는 cell에서 j번째 detector가 있을 때에만 1

  ```python
  I = iou_predict_truth 	
  max_I = tf.reduce_max(I, 2, keepdims=True)
  best_box_mask = tf.cast((I >= max_I), tf.float32
                          
  object_exists_cell = np.zeros([cell_size, cell_size, 1])
  object_exists_cell_i, object_exists_cell_j = int(cell_size * ycenter / input_height), int(cell_size * xcenter / input_width)
  object_exists_cell[object_exists_cell_i][object_exists_cell_j] = 1
                          
  noobject_indicator_function = best_box_mask * (1-object_exists_cell)
  ```

  

**contents**

- [coordinate loss]("#coordinate loss")
- [confidence loss]("#confidence loss")
- [class loss]("#class loss")
- [Total loss]("#Total loss")
- [code](#code)



---



### coordinate loss

x, y좌표와 width, height의 오차에 대해 MSE를 통해 loss를 계산한다.
$$
Coordinate\ Loss = \lambda_{coord} \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{obj}_{ij}\left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right] \\
+ \lambda_{coord} \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{obj}_{ij}\left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right] \\
$$


```python
# predict[2] == pred_coordinate	[1, cell_size, cell_size, boxes_per_cell, 4]

predict_boxes = predict[2]
predict_boxes = tf.squeeze(predict_boxes, [0])

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

coord_loss = ((tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_xcenter - xcenter) / (input_width / cell_size)) +
tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_ycenter - ycenter) / (input_height / cell_size)) + tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_w - sqrt_w)) / input_width 
+ tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_h - sqrt_h)) / input_height ) * coord_scale)
```



- width, height에 Square을 사용하는 이유: 

  object의 크기에 따라서 bounding box의 width, height의 loss 크기가 작더라도, 다른 loss에 비해 상대적으로 큰 차이처럼 영향을 미칠 수 있기 때문에 loss에 루트를 씌운다.

  > ex) 
  >
  > object 1 의 label width = 300,		 object 2 의 label width = 16
  >
  > object 1 의 prediction width = 305,		 object 2 의 prediction width = 13
  >
  > |300 - 305| =5
  >
  > |16 - 13| = 3   
  >
  > 영향은 object 1이 더 작아야 하지만, 값의 크기가  object 2에 비해 크기 때문에 이러한 부분이 학습에 반영되어 의도치 않은 학습 결과를 불러올 수 있다.





### confidence loss

`confidence loss` = `object loss` + `noobject loss`

object loss와 nobject loss는 object의 존재 여부에 따른 0 or 1의 확률을 가진 binary한 확률 예측이기 때문에 BCE를 통해 loss를 계산했다.


$$
\hat{C_i} = \sigma(C_i) = \frac{1}{1+e^{-C_i}}\\
\sigma : Sigmoid,  \ \ \ \ \ C_i : label,  \ \ \ \ \ \hat{C_i} : predicted\ value
$$

$$
Object\ Loss = \lambda_{obj} * 𝟙^{obj}_{ij} * \overline{(-[C_i*log(\hat{C_i}) + (C_i)log(1 - \hat{C_i})])} \\

NoObject\ loss =  \lambda_{noobj} * 𝟙^{noobj}_{ij}* \overline{(-[C_i*log(\hat{C_i}) + (C_i)log(1 - \hat{C_i})])}\\
$$

```python
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
    
    # predict[1] == pred_confidence [1, cell_size, cell_size, boxes_per_cell]
    pred_C = predict[1]
	pred_C = tf.squeeze(pred_C, [0])

    # object_loss
	object_loss =  tf.reduce_mean(object_exists_cell * best_box_mask * object_scale * tf.nn.sigmoid_cross_entropy_with_logits(C_label, pred_C))
		
	# noobject_loss
	noobject_loss = tf.reduce_mean((1 - object_exists_cell) * best_box_mask * tf.nn.sigmoid_cross_entropy_with_logits(C_label, pred_C) * noobject_scale)
```



- 각 cell에서 계산 한 BCE값의 평균치에 따라 backpropagation이 이루어지며 학습 될 수 있도록 `tf.reduce_mean` 을 사용했다.



### class loss

class loss는 multi class에 대한 확률 예측이기 때문에 CCE를 통해 loss를 계산했다.


$$
\hat{C_i} = Softmax(C_i) = \frac{e^{(C_i)}}{\sum_{j=1}^{K}e^{(C_j)}}\\
$$

$$
Class\ Loss = 𝟙^{obj}_{i} * \overline{(- \sum_{i=1}^{K} C_i log(\hat{C}_i))}
$$



```python
	# set class_loss information(probability, 특정 class일 확률)
    # predict[0] == pred_class 		[1, cell_size, cell_size, num_class]
	pred_P = predict[0]
	pred_P = tf.squeeze(pred_P, [0])	
	temp_P = np.zeros_like(pred_P)
	for i in range(cell_size):
		for j in range(cell_size):
				temp_P[i][j] = label[4]
	P = tf.constant(temp_P)
    
    class_loss = tf.reduce_mean(object_exists_cell * class_scale * tf.nn.softmax_cross_entropy_with_logits(P, pred_P))
```







## Total loss

**total loss** = `coordinate loss` + `confidence loss` + `class loss`
$$
Total\ loss  = 
\lambda_{coord} \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{obj}_{ij}\left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right] \\
+ \lambda_{coord} \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{obj}_{ij}\left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right] \\
+ \lambda_{obj} * 𝟙^{obj}_{ij} * \overline{(-[C_i*log(\hat{C_i}) + (C_i)log(1 - \hat{C_i})])} \\
+  \lambda_{noobj} * 𝟙^{noobj}_{ij}* \overline{(-[C_i*log(\hat{C_i}) + (C_i)log(1 - \hat{C_i})])}\\
+ 𝟙^{obj}_{i} * \overline{- \sum_{i=1}^{K} C_i log(\hat{C}_i)}
$$


### code

**description argument**

- `predict` : shape = [cell_size, cell_size, 5*boxes_per_cell + num_calsses] 인 tensor

- `labels` : 전체 label 안의 각각의 label 에는 5가지 data가 mapping되어있다. 

  >  shape = [number of object, [x coordinate, y coordinate, width, height, confidence ]]
  >
  > 각 coordinate는 normalize되지 않은 값 

- `each_object_num` : index of object

- `num_classes` : prediction하는 class의 개수

- `boxes_per_cell` : 하나의 grid cell당 예측할 bounding box

- `cell_size` 

- `input_width` : image width after resizing 

- `input_height` : image height after resizing 

- `coord_scale` : coefficients of coordinate loss

- `object_scale` : coefficients of object loss

- `noobject_scale` : coefficients of noobject loss

- `class_scale`  : coefficients of class loss



```python
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

	coord_loss = ((tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_xcenter - xcenter) / (input_width / cell_size)) +
					tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_ycenter - ycenter) / (input_height / cell_size)) +
					tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_w - sqrt_w)) / input_width +
					tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_h - sqrt_h)) / input_height ) * coord_scale)
				

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

```





**detail**

- line 71  **coord_loss** :   

  `tf.nn.l2_loss(t)` : output = sum(t ** 2) / 2

  `object_exists_cell * best_box_mask` : 𝟙^{obj}_{ij} 계산 식

  `(input_width / cell_size)` : cell을 기준으로 nomalize된 좌표 계산을 위한 식

  `input_width` : image를 기준으로 normalize로 표현

- line 82  **noobject_loss** 

  `(1 - object_exists_cell)` : object가 없는 셀에만 1의 값이 남는다.

  object가 없는 cell에서는 label confidence가 없기 때문에 `0 - pred_C`
