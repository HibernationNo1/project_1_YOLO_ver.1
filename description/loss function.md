# loss function

##### Intersection over Union

![](https://t1.daumcdn.net/cfile/tistory/993477505D14A25016)

IoU = 교집합 영역 넓이 / 합집합 영역 넓이



##### predicted bounding box

YOLO에서 best predicted bounding box 선정 기준은 모든 predicted bounding box 중에서 가장 큰 IOU 값을 가진 bounding box이다.



## Sum-Squared Error

- Square을 사용하는 이유: 

  object의 크기에 따라서 bounding box의 width, height의 loss 크기가 작더라도, 다른 loss에 비해 상대적으로 큰 차이처럼 영향을 미칠 수 있기 때문에 loss에 루트를 씌운다.

  >  ex) 
  >
  >  object 1 의 label width = 300,		 object 2 의 label width = 16
  >
  >  object 1 의 prediction width = 305,		 object 2 의 prediction width = 13
  >
  >  |300 - 305| =5
  >
  >  |16 - 13| = 3   
  >
  >  영향은 object 1이 더 작아야 하지만, 값의 크기가  object 2에 비해 크기 때문에 이러한 부분이 학습에 반영되어 의도치 않은 학습 결과를 불러올 수 있다.





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
>   단순 sum-squared error만 사용하면 object가 없는 grid cell에서는 confidence가 0이 되고, 이러한 confidence가 많아지면 학습이 불안정할 수 있기 때문에, 이를 예방하기 위해 bounding box cofidence predcition 앞에 lambda\_coord 를 곱하고, object가 없는 grid cell의 cofidence predcition 앞에는 lambda\_noodj 를 곱해준다. (가중치를 줌)
>
>   각 람다의 값은 중요도를 의미한다.
>   $$
>   \lambda_{coord} = 5, \ \ \ \ \ \lambda_{noodj} = 0.5.
>   $$
>
> - **indicator function**:
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

$$
\lambda_{coord} \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{obj}_{ij}\left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right] \\
+ \lambda_{coord} \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{obj}_{ij}\left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right]
$$





#### object loss

본 code에서는 더욱 자유로운 값의 결정을 위해 coefficient for object loss를 추가
$$
\lambda_{object} \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{obj}_{ij}(C_i - \hat{C_i})^2
$$


#### noobject loss

$$
\lambda_{noobj} \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{noobj}_{ij}(C_i - \hat{C_i})^2
$$



#### class loss

본 code에서는 더욱 자유로운 값의 결정을 위해 coefficient for class loss를 추가
$$
\lambda_{class}  \sum^{S^2}_{i = 0}𝟙^{obj}_{i}\sum_{c \in classes} (p_i(c) - \hat{p_i}(c))^2
$$




## Implement by code 

```python
import tensorflow as tf
import numpy as np
from utils import iou

def yolo_loss(predict, # [S, S, 5*B + C] dtype tensor
              labels, # object num, 5
              # 5 : x, y coordinate, width, height, confidence 
              # 각 좌표는 normalize 되지 않은 값
              each_object_num, # object의 index
              # 1개의 object에 대한 loss + 2개의 ... + n개의 object에 대한 loss = 전체 image에 대한 loss
              num_classes, # prediction하는 class의 개수
              boxes_per_cell, # 하나의 grid cell당 예측할 bounding box
              cell_size, # 몇 × 몇 cell로 나눌건지
              input_width,  # image width after resizing 
              input_height,	# image height after resizing 
              coord_scale, # lambda_coord
              object_scale, # 임의로 추가한 coefficient for object loss
              noobject_scale, # lambda_noodj
              class_scale	# 임의로 추가한 coefficient for class loss


    # parse only coordinate vector
    predict_boxes = predict[:, :, num_classes + boxes_per_cell:]
              # 5*B + C 중 앞에 (class 개수 + cell당 존재하는 box 개수) + 1 의 index부터 마지막 index까지  
              # cell_size = 7, boxes_per_cell = 2 라면  predict_boxes.shape == 7*7*8
    predict_boxes = tf.reshape(predict_boxes, [cell_size, cell_size, boxes_per_cell, 4])
              # predict_boxes.shape == 7*7*2*4

    # prediction : absolute coordinate
    pred_xcenter = predict_boxes[:, :, :, 0] # x coordinate
    pred_ycenter = predict_boxes[:, :, :, 1] # y coordinate
    pred_sqrt_w = tf.sqrt(tf.minimum(input_width * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
    pred_sqrt_h = tf.sqrt(tf.minimum(input_height * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))
              # width와 height를 (0 ~ input width) 사이의 값으로 제한하고, 루트 
    pred_sqrt_w = tf.cast(pred_sqrt_w, tf.float32)
    pred_sqrt_h = tf.cast(pred_sqrt_h, tf.float32)
              # 형 변환

    # parse label
    labels = np.array(labels)
    labels = labels.astype('float32')
    label = labels[each_object_num, :] # 전체 labels에서 하나의 label만 할당
              # label 에는 5가지 data가 mapping되어있다.
    xcenter = label[0] # x coodnate of label
    ycenter = label[1] # y coodnate of label
    sqrt_w = tf.sqrt(label[2]) # sqrt(width of label)
    sqrt_h = tf.sqrt(label[3]) # sqrt(height of label)

    # calulate iou between ground-truth and predictions
    iou_predict_truth = iou(predict_boxes, label[0:4])
              # iou_predict_truth.shape == (7, 7, 2)

    # find best box mask
    I = iou_predict_truth 
    max_I = tf.reduce_max(I, 2, keepdims=True)  # 2개의 box 중에서 iou가 높은 box만 할당
              # max_I.shape == (7, 7, 1)
    best_box_mask = tf.cast((I >= max_I), tf.float32)
              # est_box_mask.shape == (7, 7, 2) 이고, 3 dims은 0 또는 1의 값만 가지고 있다.
              # iou_predict_truth 에서 iou가 가장 높은 bounding box는 1로 대체되고, 그렇지 않은 bounding box는 0으로 대체된 것이 best_box_mask

    # set object_loss information
    C = iou_predict_truth 
    pred_C = predict[:, :, num_classes:num_classes + boxes_per_cell]

    # set class_loss information
    P = tf.one_hot(tf.cast(label[4], tf.int32), num_classes, dtype=tf.float32)
              # label[4] 에는 각각의 class에 대한 confidence 값이 들어있다. 
              # cast를 통해 int형으로 바꿔주고, 20개의 class 데이터를 one-hot encoding해준다.
    pred_P = predict[:, :, 0:num_classes]

    # find object exists cell mask
    object_exists_cell = np.zeros([cell_size, cell_size, 1])
    object_exists_cell_i, object_exists_cell_j = int(cell_size * ycenter / input_height), int(cell_size * xcenter / input_width)
              # label x, y coordinate로 전체 image 중에서 label object가 있는 cell 위치 찾기  
    object_exists_cell[object_exists_cell_i][object_exists_cell_j] = 1
              # [cell_size, cell_size] 의 각 cell 중 object가 있는 cell에만 1의 값을, 나머지는 0의 값을 가지도록 set 

    # set coord_loss
    coord_loss = (tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_xcenter - xcenter) / (input_width / cell_size)) +
                    tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_ycenter - ycenter) / (input_height / cell_size)) +
                    tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_w - sqrt_w)) / input_width +
                    tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_h - sqrt_h)) / input_height ) 
                * coord_scale
              # 𝟙^{obj}_{ij} 를 object_exists_cell * best_box_mask로 표현, coord_scale == lambda_coord
              # (input_width / cell_size) : cell을 기준으로 nomalize로 좌표 표현
              # input_width : image를 기준으로 normalize로 표현
              
	
    # object_loss
    object_loss = tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_C - C)) * object_scale

    # noobject_loss
    noobject_loss = tf.nn.l2_loss((1 - object_exists_cell) * (pred_C)) * noobject_scale
              # (1 - object_exists_cell) : object가 없는 셀에만 1의 값이 남는다.
              # object가 없는 cell에서는 label confidence가 없기 때문에 0 - pred_C

    # class loss
    class_loss = tf.nn.l2_loss(object_exists_cell * (pred_P - P)) * class_scale

    # sum every loss
    total_loss = coord_loss + object_loss + noobject_loss + class_loss

    return total_loss, coord_loss, object_loss, noobject_loss, class_los
```





