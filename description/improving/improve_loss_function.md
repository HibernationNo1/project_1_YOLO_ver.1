**loss_function의 계산 과정을 YOLO V3의 loss 계산 방법을 참고해서 수정했다.**



### **todo list**

- modify class loss 

  1. [ `tf.one_hot`의 argument 변경](#'modify compute method for P')

  2. [ `pred_P`에 activation function 적용](#'modify compute method for pred_P')

  3. [modify class loss function](#class_loss_function)

- modify object loss



### Improving

#### class_loss

###### modify compute method for P

loss_function에서 label값으로 사용되는 `P`에 `tf.one_hot`이 적용되는데, 

이 때 `tf.one_hot`의 첫 번째 argument는 모든 label class가 할당되어야 한다.

하지만 `label[4]`는 특정 단일 class의 label number값을 가지고 있어 반환값이 [0, 0] 으로 나오는 것을 확인했다.

(tf.one_hot은 단일 label에 적용되는것이 아닌, 전체 label data에 적용되어야 한다고 알고있음)

> num_class가 2일 때 `P` 의 값은 [0, 1] 또는 [1, 0]의 값이,
>
> num_class가 3일 때 `P` 의 값은 [0, 0, 1] 또는 [0, 1, 0] 또는 [1, 0, 0]의 값이 사용되어야 한다고 생각한다.

제대로 된 one_hot encoding을 위해

각 label값에 알맞게 one-hot encoding 된 값이 할당될 수 있도록 function을 추가했다.

- `tf.one_hot`의 첫 번째 argument에 `num_classes` 만큼의 count number를 가진 list를 사용

  ```python
  # loss.py
  def class_loss_one_hot(num_classes):
  	index_list = [i for i in range(num_classes)]
  	P_one_hot = (tf.one_hot(tf.cast((index_list), tf.int32), num_classes, dtype=tf.float32))
  	return P_one_hot
  ```

  해당 function은 loss.py에서 정의했지만, 사용시 학습 속도가 저하되지 않기 위해 가장 상위 계층인 main.py의 global에 instance를 선언했다.

  ```python
  # main.py
  from loss import class_loss_one_hot
  P_one_hot = class_loss_one_hot(class_name_dict.keys())
  ```

  

- `label[4]`값에 알맞게 one_hot encoding이 적용된 정답값을 P에 할당

  ```python
  # loss.py  def yolo_loss
  for i in range(num_classes):
  	if label[4] == list(class_name_dict.keys())[i]:
  		P = P_one_hot[i]
  ```

  

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

  

변경 전 `P`

```python
P = tf.one_hot(tf.cast(label[4], tf.int32), num_classes, dtype=tf.float32)

```



변경 후 `P`

````python
p = 0.0
for i in range(num_classes):
	if label[4] == list(class_name_dict.keys())[i]:
		P = P_one_hot[i]
````





###### modify compute method for pred_P

`pred_P`의 값은 multi class에 대한 probability를 표현해야 하는데 그런 과정이 생략되어 있음을 확인했다.

`pred_P`는 각 cell마다 n개의 class에 대한 각각의 probability를 가지고 있어야 한다. (그 합은 1)

이를 위해 softmax activation function을 적용했다.

>  다수의 calss에 대한 probabiliy를 표현하기 때문에 softmax activation function을 사용하도록 한다.



변경 전 `pred_P`

```python
pred_P = predict[:, :, 0:num_classes] 
```



변경 후 `pred_P`

```python
pred_P = predict[:, :, 0:num_classes] 
temp_pred_P = np.zeros_like(pred_P)
for i in range(cell_size):
		for j in range(cell_size):
			temp_pred_P[i][j] = tf.nn.softmax(pred_P[i][j]) 
pred_P = tf.constant(temp_pred_P)
```





###### class_loss_function

class loss는 예측한 특정 class에 대한 probability를 표현하기 때문에 사용되는 loss functiond은 MSE가 아닌 CategoricalCrossentropy가 적절하다고 판단했다.

변경 전 `class_loss`

```python
class_loss = tf.nn.l2_loss(object_exists_cell * (pred_P - P)) * class_scale
```



변경 후 `class_loss`

```python
class_loss = object_exists_cell * class_scale * class_loss_object(P, pred_P)
```

