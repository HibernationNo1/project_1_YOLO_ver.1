# Analysis

training은 10번의 epoch가 진행 될 때 까지 수행했으며, 그 과정에서 Validation은 `step%100 == 0`마다 수행되도록 진행했습니다.



- [loss](#loss)
  - [train loss](#train-loss)
  - [Validatio loss](#Validation-loss)
  - [conclusion](#conclusion)
- [check result with validation, test image](#check-result-with-validation,-test image)
- [Check learning rate decay](#Check-learning-rate-decay)



## loss

#### train loss

- class_loss 

  ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/main_calss_loss.png?raw=true)

- coord_loss 

  ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/main_coord_loss.png?raw=true)

- noobject_loss

  ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/main_noobject_loss.png?raw=true)

- object_loss 

  ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/main_object_loss.png?raw=true)

  

**total_loss**

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/main_total_loss.png?raw=true)



#### Validation loss

Validation loss는 10개의 image에 대한 loss의 총 합을 표현했기 때문에 test loss보다 10배 큰 값으로 기록되었다.

더욱 낮은 Validation loss을 위해 drop out의 비율을 0.2, 0.3, 0.4를 시도하고 kernel_regularizer 의 값을 각각의 dense layer에 대해서 coordinate dense L1 = 0.02~0.1 , class dense L2 = 0.01~0.05 , confidence dense L2 = 0.01~0.03를 적용해 보았지만 학습 간 의미있는 Validation loss의 최소값 변화는 없었다.

이를 통해 overfitting 문제는 없음을 확인했다.

- class_loss : 

  ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/validation_calss_loss.png?raw=true)

- coord_loss : 

  ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/validation_coord_loss.png)

- noobject_loss : 

  ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/validation_noobject_loss.png?raw=true)

- object_loss : 

  ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/validation_object_loss.png?raw=true)



**total loss**

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/validation_total_loss.png?raw=true)



### conclusion

400번째 step부터 loss의 큰 변화 없이 1epoch(약 120step)마다 일정한 파형을 그리지만, 1epoch의 평균 최소값은 작아지지 않음을 보여준다. 이에 따라 intersection of union값 역시 커지지 않으며 학습의 진전이 없는 것으로 판단된다.



더 나은 학습을 위한 시도(진행중)

- input image의 width와 height를 224에서 448로 증가시켜도 학습의 정확성은 증가하지 않고, loss만 두 배로 커짐을 확인했다.

- learning rate의 감소율을 400step부터 더욱 크게 증가시켜도 파형이 유지되는 step이 많아질 뿐, 파형 자체는 유지됨을 확인했다.

- YOLO model에 사용되었던 tf.keras.applicatio inceptionV3대신 각 layer을 직접 쌓는 식으로 model을 구현해 보았지만 loss가 소폭 증가할 뿐 개선되지는 않았다.

   



## check result with validation, test image

The right is the label image and the left is the predicted image by model

**validation image**

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/val1.png?raw=true)

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/val2.png?raw=true)

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/val5.png?raw=true)





**test image**

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/test1.png?raw=true)

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/test2.png?raw=true)

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/94_result.png?raw=true)



backbone network로 darknet을 사용한 YOLO model에 비해 전체적으로 object detection rate는 크게 낮은 결과를 보여주고 있으며 다중 object에 대해서 detection 비율은 좋지 못함을 볼 수 있다.

특히, cat은 object의 color가 background color와 크게 다르고 귀의 모양이 두드러지게 확인이 가능할 때 더욱 높은 detect rate를 보여주었고, horse는 다리와 머리가 모두 측면으로 나타나는 image에서 높은 detect rate를 노여주는 것으로 확인되었다.



- confidence score가 0.7 이상인 object만을 검출했을 때 performance evaluation에 의한 average detection rate는 20%를 넘지 못하는 낮은 수치를 확인할 수 있었다.

  정답과 같은 수의 object를 detection한 perfect detection인 경우 해당 class에 대해서 정답과 같은 class를 예측한 classification accuracy는 100%임을 확인했다.

- confidence score가 0.5 이상인 object만을 검출했을 때 실제 object보다 더 많은 object를 검출하는 경우인 over detect인 경우가 크게 높아져 average detection rate가 여전히 20%를 넘지 못했다.

  이를 통해 가장 이상적인 경우는 confidence score가 1에 가까운 경우일 때 Bbox를 표현하는 것이고, 이를 위해서는 coordinate loss를 줄이는 방향으로 개선해야 한다는 것을 알 수 있었다.



## Check learning rate decay

flags를 통해 initial learning rate는 0.0001로,  학습 과정에서 100번의 step마다 learning rate가 0.5씩 곱해져 적용되도록 해 놓았습니다.

```python
flags.DEFINE_float('init_learning_rate', default=0.0001, help='initial learning rate') # original paper : 0.001 (1epoch) -> 0.01 (75epoch) -> 0.001 (30epoch) -> 0.0001 (30epoch)

flags.DEFINE_float('lr_decay_rate', default=0.5, help='decay rate for the learning rate')

flags.DEFINE_integer('lr_decay_steps', default=100, help='number of steps after which the learning rate is decayed by decay rate') # 2000번 마다 init_learning_rate * lr_decay_rate 을 실행
```



그래프를 통해 learning rate의 decay가 잘 적용되었는지 확인해보자

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/learning%20rate.png?raw=true)

학습은 총 1k step까지 진행이 되었고, 매 100번째 step마다 0.5의 값이 learning rate에 곱해져 적용되었음을 확인할 수 있습니다.
