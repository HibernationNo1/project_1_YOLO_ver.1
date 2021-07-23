**class_loss의 계산 과정에서 잘못된 계산이 진행됨을 발견, 계산 과정을 수정했다.**



loss.py 의 `yolo_loss` function에서 class loss를 구하는 과정에 오류를 발견했다.

1. MSE에 사용되는 값 중 label값으로 사용되는 `P`에 `tf.one_hot`이 적용되는데, 

   이 때 `tf.one_hot`의 첫 번째 argument는 모든 label class가 할당되어야 한다.

   하지만 `label[4]`는 특정 단일 class의 label number값을 가지고 있어 반환값이 [0, 0] 으로 나오는 것을 확인했다.

   (tf.one_hot은 단일 label에 적용되는것이 아닌, 전체 label data에 적용되어야 한다고 알고있음)

   > num_class가 2일 때 `P` 의 값은 [0, 1] 또는 [1, 0]의 값이,
   >
   > num_class가 3일 때 `P` 의 값은 [0, 0, 1] 또는 [0, 1, 0] 또는 [1, 0, 0]의 값이 사용되어야 한다고 생각한다.

2. MSE에 사용되는 값 중 prediction값으로 사용되는 `pred_P`의 각 element는 그 합이 1이여야 하는데

   (각 calss에 대한 probability를 표현하기 때문), 각각의 element의 절대값이 1을 넘어가는(probability가 아닌) 것을 확인했다.

   `pred_P[:, :, 0]`

   ```
   pred_P:  tf.Tensor(
   [[[ -9.340251    -0.9598994 ]
     [ -9.862955     8.061177  ]
     [ 12.465103     5.901286  ]
     [ -4.9818554   -7.560293  ]
     [  4.3412714   11.126848  ]
     [  7.535841    -3.9240193 ]
     [  4.247219    -5.378591  ]]
   
    [[-17.439072    -1.5861354 ]
     [ 19.343       -8.920415  ]
     [  1.1232624  -11.820934  ]
     [  6.421815     7.532135  ]
     [ -3.863473   -22.316135  ]
     [-18.011848    -2.160532  ]
     [ -0.9425481    7.575761  ]]
   
    [[  1.1111486    9.589155  ]
     [  9.118893   -26.612255  ]
     [ 10.1727705   -7.8699875 ]
     [ 10.630701     2.231781  ]
     [  2.0918865   -8.619567  ]
     [  7.6819067   16.852806  ]
     [ 19.056692    -0.17274404]]
   
    [[  1.3355023    1.9120104 ]
     [-14.368527     0.804693  ]
     [ -0.12353039  10.682286  ]
     [-23.10432      5.925239  ]
     [ 10.189813     2.6157815 ]
     [  6.996082     0.87901175]
     [ -3.7928183   -0.98149997]]
   
    [[ -1.7510042   25.113008  ]
     [  9.673466    -5.6224093 ]
     [  1.5459926  -24.09742   ]
     [-10.630903     2.6801858 ]
     [ -5.420775    11.139401  ]
     [  2.9250698   16.096392  ]
     [ 16.104355    10.893917  ]]
   
    [[ -5.9980135  -10.349937  ]
     [  6.2316914    0.45182776]
     [ 12.446317     5.8213644 ]
     [ 11.106987    -0.4601066 ]
     [-12.386986     5.170607  ]
     [ -8.38606      1.9685433 ]
     [ 15.304098    23.250185  ]]
   
    [[  4.1113305    6.762926  ]
     [-10.502751    22.056528  ]
     [ -3.661048   -12.553807  ]
     [ 10.809967     7.508924  ]
     [ 13.913809     5.8134346 ]
     [-11.604847     8.364764  ]
     [  1.067491     7.5060124 ]]], shape=(7, 7, 2), dtype=float32)
   ```

   

### Improving



#### fix compute method for P

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



#### fix compute method for pred_P

`pred_P`는 각 cell마다 n개의 class에 대한 각각의 probability를 가지고 있어야 한다. (그 합은 1)

- softmax를 위한 model을 하나 더 만들어서 `pred_P`에 할당된 node들만 따로 학습시켜보자.

  pred_P만 `flatten()`으로 만든 후 `Dense(cell_size * cell_size * num_classes , activation = 'softmax')`

  예상 결과 : (cell_size * cell_size * num_classes)개 노드의 합이 1이 되어버린다. 내가 원하는 것은 각 cell마다 num_classes node의 합이 1이 되는 것이다.

  

- pred_P 값이 확률이 아니니까, [1, 0] [0, 1] 을 구분해서 학습하기 어려울 것이다.

  두 값을 구분하기 쉽도록 [10, 0] [0, 10]으로 바꿔보자.

  예상 결과 : class loss function의 기울기가 더욱 커질것이다.

   partial derevative는 function의 변화량이 증가하는 방향임으로 학습에 더욱 도움이 되지 않을까?

  ```
  label[4] : 7.0,    P: [10. 0.]
  label[4] : 7.0,    P: [10. 0.]
  label[4] : 9.0,    P: [0. 10.]
  label[4] : 7.0,    P: [10. 0.]
  label[4] : 9.0,    P: [0. 10.]
  label[4] : 7.0,    P: [10. 0.]
  label[4] : 7.0,    P: [10. 0.]
  label[4] : 7.0,    P: [10. 0.]
  label[4] : 7.0,    P: [10. 0.]
  label[4] : 7.0,    P: [10. 0.]
  label[4] : 9.0,    P: [0. 10.]
  label[4] : 9.0,    P: [0. 10.]
  ```

  학습 결과

  class_loss가 더 잘 안떨어지는 모습을 보였다.

