**loss_function의 계산 과정을 YOLO V3의 loss 계산 방법을 참고해서 수정했다.**



- 기존 code의 `loss_function`에서 label값으로 사용되는 `P`에 `tf.one_hot`이 적용되고,  `tf.one_hot`의 첫 번째 argument는  `label[4]`가 사용된다.

  이 때 `tf.one_hot`의 첫 번째 argument는 모든 label class가 할당되어야 하지만 `label[4]`는 특정 단일 class의 label number값을 가지고 있어 반환값이 [0, 0] 으로 나오는 것을 확인했다.

  (tf.one_hot은 단일 label에 적용되는것이 아닌, 전체 label data에 적용되어야 한다.)

  >num_class가 2일 때 `P` 의 값은 [0, 1] 또는 [1, 0]의 값이,
  >
  >num_class가 3일 때 `P` 의 값은 [0, 0, 1] 또는 [0, 1, 0] 또는 [1, 0, 0]의 값이 사용되어야 한다고 생각한다.





### **todo list**

- one_hot encoding

  



### Improving

#### add one hot encoding

제대로 된 one_hot encoding을 위해

각 label값에 알맞게 one-hot encoding이 적용된 값이 할당될 수 있도록 dataset.py의 `process_each_ground_truth`에 one-hot encoding 처리 코드를 추가했다.

- `label[4]`값에 알맞게 one_hot encoding이 적용된 정답값을 P에 할당

  ```python
  class_num = class_labels[i] # 실제 class labels
  
  # ont_hot preprocess
  num_of_class = len(class_name_dict.keys()) 
  index_list = [n for n in range(num_of_class)]
  oh_class_num = (tf.one_hot(tf.cast((index_list), tf.int32), num_of_class, dtype=tf.float32))
  for j in range(num_of_class): 
  	if int(class_num) == list(class_name_dict.keys())[j]:
  		class_num = oh_class_num[j]
  			break
  ```
  
  `tf.one_hot`의 첫 번째 argument에 `num_classes` 만큼의 count number를 가진 list를 사용
  
  
  
  **결과**
  
  ```
  label[4] : 7.0,    P: [1. 0.]
  label[4] : 7.0,    P: [1. 0.]
  label[4] : 9.0,    P: [0. 1.]
  label[4] : 7.0,    P: [1. 0.]
  label[4] : 9.0,    P: [0. 1.]
  label[4] : 7.0,    P: [1. 0.]
  label[4] : 7.0,    P: [1. 0.]
  label[4] : 7.0,    P: [1. 0.]
  label[4] : 7.0,    P: [1. 0.]
  label[4] : 7.0,    P: [1. 0.]
  label[4] : 9.0,    P: [0. 1.]
  label[4] : 9.0,    P: [0. 1.]
  ```
  
  두 개의 class가 있고, 첫 번째 class의 label은 7이고 두 번째 class의 label은 9일 때
  
  `label[4]` == 7.0 일땐  `P`의 값이 [1. 0.] 이고,  `label[4]` == 9.0 일땐  `P`의 값이 [0. 1.] 임을 확인 
  



`yolo_loss function` 에서도 `P`값의 code수정

**변경 전 **

```python
I = iou_predict_truth 
max_I = tf.reduce_max(I, 2, keepdims=True)
best_box_mask = tf.cast((I >= max_I), tf.float32) 
```

> I = 0 이면 best_box_mask의 모든 element가 1이 된다.



**변경 후 **

```python
temp_P = np.zeros_like(pred_C)
for i in range(cell_size):
	for j in range(cell_size):
			temp_P[i][j] = label[4]
P = tf.constant(temp_P)
```

> label은 scala이기 때문에 loss function의 input pred_P와 맞게 shape을 만들어줘야 한다.



#### class_loss

##### modify `pred_C`, `pred_P`

model.py에서 각 units를 학습 목적에 맞게 분류하고 activation function을 적용했기 때문에 loss.py의 `pred_c`와 `pred_P`의 호출 과정을 수정했다.



**변경 전 `pred_C`**

```python
pred_C = predict[:, :, num_classes:num_classes + boxes_per_cell]
```

**변경 후 `pred_C`**

```python
pred_C = predict[1]
pred_C = tf.squeeze(pred_C)
```



**변경 전 `pred_P`**

```python
pred_P = predict[:, :, 0:num_classes] 
```

**변경 후 `pred_P`**

```python
pred_P = predict[0]
pred_P = tf.squeeze(pred_P)
```





##### modify `C`

confidence loss의 label값 `C`는 label Bbox와 predicted value간의 interception union을 사용했다.

하지만 initial prediction value는 학습이 되지 않은 값이기 때문에 0에 가까운 값을 반환한다. 

그렇기 때문에 학습의 처음부터 잘못된 loss값이 사용되는 것이다.

이상적인 학습을 위한 confidence loss의 label값은 1이다. indicator function에 의해 object가 있는 cell에 대해서 confidence value는 반드시 1이기 때문이다.



**변경 전 `c`**

```python
C = iou_predict_truth
```

**변경 후 `c`**

```python
C = 1
```





##### class_loss_function

class loss는 예측한 특정 class에 대한 probability를 표현하기 때문에 사용되는 loss functiond은 MSE가 아닌 CategoricalCrossentropy가 적절하다고 판단했다.

변경 전 `class_loss`

```python
class_loss = tf.nn.l2_loss(object_exists_cell * (pred_P - P)) * class_scale
```



변경 후 `class_loss`

```python
class_loss = tf.reduce_sum(object_exists_cell * class_scale * class_loss_object(P, pred_P))
```

> class_loss_object는 train.py에서 선언된 CategoricalCrossentropy의 object이다.





##### confidence_loss_function

object_loss와 noobject_loss의 값의 범위가 sigmoid로 인해 0~1이기 때문에 loss function으로 MSE보다 **BCE**를 사용하는게 옳은 방법이다



**변경 전 `confidence_loss`**

```python
object_loss = tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_C - C)) * object_scale
noobject_loss = tf.nn.l2_loss((1 - object_exists_cell) * (pred_C)) * noobject_scale
confidence_loss = object_loss + noobject_loss
```

**변경 후 `confidence_loss`**

```python
object_loss = tf.reduce_sum(object_exists_cell * best_box_mask * confidence_loss_object(C, pred_C) * object_scale)

noobject_loss = tf.reduce_sum((1 - object_exists_cell) * confidence_loss_object(0, pred_C) * noobject_scale)
```

> confidence_loss_object는 train.py에서 선언된 BinaryCrossentropy의 object이다.

