# loss function.py

model의 loss function을 정의한다.



total loss = coordinate loss + object loss + noobject loss + class loss

coordinate loss와 object loss는 best predicted bounding box에 대해서만 표현되기 때문에 code상에서는 모든 bounding box 중에서 가장 높은 IOU가 곱해지도록 구현했다.



##### Intersection over Union(IOU)

![img](https://t1.daumcdn.net/cfile/tistory/993477505D14A25016)

IoU = 교집합 영역 넓이 / 합집합 영역 넓이

best predicted bounding box란 모든 predicted bounding box 중에서 가장 큰 IOU 값을 가진 bounding box이다. 



**define function**

- [yolo_loss](#yolo_loss)



## Sum-Squared Error

- Square을 사용하는 이유: 

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



### Total loss

Total loss = coordinate loss + object loss + noobject loss + class loss

**수식**

$$
\lambda_{coord} \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{obj}_{ij}\left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right] \\
+ \lambda_{coord} \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{obj}_{ij}\left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right] \\ 
+ \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{obj}_{ij}(C_i - \hat{C_i})^2\\ 
+ \lambda_{noobj} \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{noobj}_{ij}(C_i - \hat{C_i})^2\\ 
+ \sum^{S^2}_{i = 0}𝟙^{obj}_{i}\sum_{c \in classes} (p_i(c) - \hat{p_i}(c))^2
$$


> - **x** : object의 x좌표(grid 기준)
>
> - **y** : object의 y좌표(grid 기준)
>
> - **i** : i번째 grid cell
>
> - **j** : j 번째 detector
>
> - **w** : bounding box의 width(전체 이미지 기준)
>
> - **h** : bounding box의 height(전체 이미지 기준)
>
> - **lambda** :
>
>   단순 sum-squared error만 사용하면 object가 없는 grid cell에서는 confidence가 0이 되고, 이러한 confidence가 많아지면 학습이 불안정할 수 있기 때문에, 이를 예방하기 위해 bounding box cofidence predcition 앞에 lambda_coord 를 곱하고, object가 없는 grid cell의 cofidence predcition 앞에는 lambda_noodj 를 곱해준다. (가중치를 줌)
>
>   각 람다의 값은 중요도를 의미한다.
>
> $$
> \lambda_{coord} = 5, \ \ \ \ \ \lambda_{noodj} = 0.5.
> $$
>
> 
>
> - **indicator function**:
>
>   i : i번째 grid cell (0부터 S^2까지)
>
>   j : j 번째 detector (0부터 Bboxs per cell)
>   
>   특정 grid cell 중에서 믿을만한 bounding box만 살리고 나머진 버리는 용도
>
>   - $$
>     𝟙^{obj}_{ij}
>     $$
>
>     i 번째 grid cell에 object가 있고, 해당 cell 안에 j번째 detector가 있을 때에만 1을 return. 그 외에는 0
>   
>     > object가 있는 cell에서 j번째 detector가 있을 때에만 1
>
>   - $$
>     𝟙^{noobj}_{ij}
>     $$
>   
>     i 번째 grid cell에 object가 없고, 해당 cell 안에 j번째 detector가 있을 때에만 1을 return. 그 외에는 0
>   
>     > object가 없는 cell에서 j번째 detector가 있을 때에만 1





#### coordinate loss

bounding box의 정보(x, y좌표, width, height)에 대한 loss값이다.


$$
\lambda_{coord} \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{obj}_{ij}\left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right] \\
+ \lambda_{coord} \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{obj}_{ij}\left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right]
$$



**indicator function**

object가 있는 cell에서 2개의 detector 중 j번째 detector가 responsible이면 1
$$
𝟙^{obj}_{ij}
$$


#### object loss

object가 있는 cell의 confidence loss.

label confidence는 ground truth와 proposed region간의 IOU를 사용한다.

본 code에서는 더욱 자유로운 값의 결정을 위해 coefficient for object loss를 추가

$$
\lambda_{object} \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{obj}_{ij}(C_i - \hat{C_i})^2
$$



#### noobject loss

object가 없는 cell의 confidence loss를 계산.

label confidence는 ground truth와 proposed region간의 IOU를 사용하기 때문에 값이 0이다.
$$
\lambda_{noobj} \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{noobj}_{ij}(\hat{C_i})^2
$$

**indicator function**

object가 없는 cell에서 2개의 detector 중 j번째 detector가 responsible이면 1
$$
𝟙^{noobj}_{ij}
$$




#### class loss

label class와 predicted class의 probability loss를 계산

본 code에서는 더욱 자유로운 값의 결정을 위해 coefficient for class loss를 추가

$$
\lambda_{class}  \sum^{S^2}_{i = 0}𝟙^{obj}_{i}\sum_{c \in classes} (p_i(c) - \hat{p_i}(c))^2
$$

**indicator function**

 detector와 상관 없이 object가 있는 cell일때만 1
$$
𝟙^{obj}_{i}
$$





## Implement by code 



**import** 

```python
 import tensorflow as tf
 import numpy as np
 from utils import iou
```





### yolo_loss.py

1개의 object에 대한 loss + 2개의 ... + n개의 object에 대한 loss = 전체 image에 대한 loss

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
               num_classes,
               boxes_per_cell,
               cell_size,
               input_width,
               input_height,
               coord_scale,
               object_scale,
               noobject_scale,
               class_scale
               ):
 
 # parse only coordinate vector
     
     # 5*B + C 중 앞의 ((class 개수 + cell당 존재하는 box 개수) + 1 의 index)부터 (마지막 index)까지 extraction 
     predict_boxes = predict[:, :, num_classes + boxes_per_cell:]
     # predict_boxes의 shape 
     # [0, 1, ... , ~num_class, 첫 번째 Bbox의 confidence 예측값, 두 번째 Bbox의 confidence 예측값, 첫 번째 Bbox의 x, y, w, h, 두 번째 Bbox의 x, y, w, h  ]
     
     # cell_size = 7, boxes_per_cell = 2 일 때 predict_boxes의 shape을 7*7*(2×4) 로 reshape해준다.
     predict_boxes = tf.reshape(predict_boxes, [cell_size, cell_size, boxes_per_cell, 4])
 
 # prediction : absolute coordinate
     pred_xcenter = predict_boxes[:, :, :, 0]
     pred_ycenter = predict_boxes[:, :, :, 1]
     # width와 height를 (0 ~ input width) 사이의 값으로 제한하고, sqrt를 적용
     pred_sqrt_w = tf.sqrt(tf.minimum(input_width * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
     pred_sqrt_h = tf.sqrt(tf.minimum(input_height * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))
     pred_sqrt_w = tf.cast(pred_sqrt_w, tf.float32)
     pred_sqrt_h = tf.cast(pred_sqrt_h, tf.float32)
 
 # parse label
     labels = np.array(labels)
     labels = labels.astype('float32')
     label = labels[each_object_num, :]      # 전체 labels에서 하나의 label만 할당
     xcenter = label[0]      # x coodnate of label
     ycenter = label[1]      # y coodnate of label
     sqrt_w = tf.sqrt(label[2])  # sqrt(width of label)
     sqrt_h = tf.sqrt(label[3])  # sqrt(height of label)
 
 # calulate iou between ground-truth and predictions
     # iou_predict_truth.shape == (cell_size, cell_size, boundingboxes_per_cell)
     iou_predict_truth = iou(predict_boxes, label[0:4])
 
 # find best box mask
     I = iou_predict_truth
     max_I = tf.reduce_max(I, 2, keepdims=True) # 각 cell 별로 2개의 box 중에서 iou가 높은 box만 할당
     # I가 max_I보다 크거나 같으면 1, 작으면 0 할당 → (7, 7, 2)을 (7*7*1)의 채널이 두 개 있다고 가정하면 IOU가 높은 채널에만 1을, 아닌 채널에는 0을 할당해서 7*7의 mask iamge가 두 개 만들어진다.
     # 이 때 1인 cell은 object cell, 0인 cell은 noobject cell이다.
     best_box_mask = tf.cast((I >= max_I), tf.float32)
 
 # set object_loss information
     C = iou_predict_truth
     pred_C = predict[:, :, num_classes:num_classes + boxes_per_cell]
 
 # set class_loss information
     # label[4] 에는 각각의 class에 대한 confidence값이 들어있다. 
     # cast를 통해 int형으로 바꿔주고, 20개의 class data를 one-hot encoding해준다.
     P = tf.one_hot(tf.cast(label[4], tf.int32), num_classes, dtype=tf.float32)
     pred_P = predict[:, :, 0:num_classes]
 
 # find object exists cell mask
     object_exists_cell = np.zeros([cell_size, cell_size, 1])
     # label의 x, y coordinate로 전체 image 중에서 label object가 있는 cell 위치 찾기 
     # [cell_size, cell_size] 의 각 cell 중 object가 있는 cell에만 1의 값을, 나머지는 0의 값을 가지도록 set (label mask image 만듦)
     object_exists_cell_i, object_exists_cell_j = int(cell_size * ycenter / input_height), int(cell_size * xcenter / input_width)
     object_exists_cell[object_exists_cell_i][object_exists_cell_j] = 1
 
 # coord_loss
     coord_loss = (tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_xcenter - xcenter) / (input_width / cell_size)) +
                     tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_ycenter - ycenter) / (input_height / cell_size)) +
                     tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_w - sqrt_w)) / input_width +
                     tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_h - sqrt_h)) / input_height ) 
                 * coord_scale
 
 # object_loss
     object_loss = tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_C - C)) * object_scale
 
 # noobject_loss
     noobject_loss = tf.nn.l2_loss((1 - object_exists_cell) * (pred_C)) * noobject_scale
 
 # class loss
     class_loss = tf.nn.l2_loss(object_exists_cell * (pred_P - P)) * class_scale
 
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
