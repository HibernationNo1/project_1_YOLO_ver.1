**loss_function의 계산 과정을 YOLO V3의 loss 계산 방법을 참고해서 수정했다.**



### **todo list**

- modify class loss 

  1. [ `tf.one_hot`의 argument 변경](#'modify compute method for P')

  2. [ `pred_P`에 activation function 적용](#'modify compute method for pred_P')

  3. [modify class loss function](#class_loss_function)
- modify object loss
  1. label값 변경
  2. `pred_C`에 activation function 적용
  3. modify class loss function



### Improving

#### class_loss

##### modify compute method for P

loss_function에서 label값으로 사용되는 `P`에 `tf.one_hot`이 적용되는데, 

이 때 `tf.one_hot`의 첫 번째 argument는 모든 label class가 할당되어야 한다.

하지만 `label[4]`는 특정 단일 class의 label number값을 가지고 있어 반환값이 [0, 0] 으로 나오는 것을 확인했다.

(tf.one_hot은 단일 label에 적용되는것이 아닌, 전체 label data에 적용되어야 한다고 알고있음)

> num_class가 2일 때 `P` 의 값은 [0, 1] 또는 [1, 0]의 값이,
>
> num_class가 3일 때 `P` 의 값은 [0, 0, 1] 또는 [0, 1, 0] 또는 [1, 0, 0]의 값이 사용되어야 한다고 생각한다.

제대로 된 one_hot encoding을 위해

각 label값에 알맞게 one-hot encoding 된 값이 할당될 수 있도록 dataset.py의 `process_each_ground_truth`에 one-hot encoding 처리 코드를 추가했다.

- `label[4]`값에 알맞게 one_hot encoding이 적용된 정답값을 P에 할당

  ```python
  class_num = class_labels[i] # 실제 class labels
  
  num_of_class = len(class_name_dict.keys()) 
  index_list = [i for i in range()]
  oh_class_num = (tf.one_hot(tf.cast((index_list), tf.int32), num_of_class, dtype=tf.float32))
  if class_num == list(class_name_dict.keys())[i]:
  	class_num = oh_class_num[i]
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
  



`yolo_loss function` 에서도 `label`과 `P`값의 code수정

**변경 전 **

```python
labels = np.array(labels) 
labels = labels.astype('float32')
```



```python
P = tf.one_hot(tf.cast(label[4], tf.int32), num_classes, dtype=tf.float32)
```



**변경 후 **

```python
	labels = np.array(labels) 
	for i in range(4):	
		labels[:, i] = labels[:, i].astype('float32')
	label = labels[each_object_num, :]

```

> tf.shape(labels) == [1, 1, 1, 1, [2]] 이기 때문에 for문 사용



````python
P = label[4]
````





##### modify compute method for pred_P

`pred_P`의 값은 multi class에 대한 probability를 표현해야 하는데 그런 과정이 생략되어 있음을 확인했다.

`pred_P`는 각 cell마다 n개의 class에 대한 각각의 probability를 가지고 있어야 한다. (그 합은 1)

이를 위해 softmax activation function을 적용했다.

>  다수의 calss에 대한 probabiliy를 표현하기 때문에 softmax activation function을 사용하도록 한다.



**변경 전 `pred_P`**

```python
pred_P = predict[:, :, 0:num_classes] 
```



**변경 후 `pred_P`**

```python
pred_P = predict[:, :, 0:num_classes] 
temp_pred_P = np.zeros_like(pred_P)
for i in range(cell_size):
		for j in range(cell_size):
			temp_pred_P[i][j] = tf.nn.softmax(pred_P[i][j]) 
pred_P = tf.constant(temp_pred_P)
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



###  confidence_loss

##### modify label value of confidence loss

confidence loss의 label값은 `C`는 label Bbox와 predicted value간의 interception union을 사용했다.

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





##### modify predicted value of confidence loss

confidence loss = object_loss + noobject_loss이고, object loss에서 사용하는 label은 1,  noobject_loss에서 사용하는 label은 0이다.
그리고 object_loss와 noobject_loss의 값의 범위는 0~1이 나와야 한다.(object가 있는지에 대한 probability이기때문이다.)
그렇기 때문에 pred_C는 sigmoid function이 적용되어야 한다.

이에 따라 pred_C의 계산 방법을 수정했다.



**변경 전 `pred_C`**

```python
pred_C = predict[:, :, num_classes:num_classes + boxes_per_cell]
```



**변경 후 `pred_C`**

```python
pred_C = predict[:, :, num_classes:num_classes + boxes_per_cell]
temp_pred_C = np.zeros_like(pred_C)
for i in range(cell_size):
		for j in range(cell_size):
			temp_pred_C[i][j] = tf.sigmoid(pred_C[i][j]) 
pred_C = tf.constant(temp_pred_C)
```



##### confidence_loss_function

object_loss와 noobject_loss의 값의 범위가 sigmoid로 인해 0~1이기 때문에 loss function으로 MSE보다 **BCE**를 사용하는게 옳은 방법이다

(sigmoid function는 베르누이 분포를 상정하기 때문이다.)



**변경 전 confidence loss function**

```python
object_loss = tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_C - C)) * object_scale
noobject_loss = tf.nn.l2_loss((1 - object_exists_cell) * (pred_C)) * noobject_scale
confidence_loss = object_loss + noobject_loss
```



**변경 후 confidence loss function**

```python
object_loss = tf.reduce_sum(object_exists_cell * best_box_mask * confidence_loss_object(C, pred_C) * object_scale)

noobject_loss = tf.reduce_sum((1 - object_exists_cell) * confidence_loss_object(0, pred_C) * noobject_scale)
```

> confidence_loss_object는 train.py에서 선언된 BinaryCrossentropy의 object이다.
