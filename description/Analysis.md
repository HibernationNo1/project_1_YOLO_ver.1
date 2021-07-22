# Analysis

training은 10번의 epoch가 진행 될 때 까지 수행했으며, 그 과정에서 Validation은 `step%50 == 0`마다 수행되도록 진행했다.



## Train loss

- class_loss : loss of predicted class value

  ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/class_loss.jpg?raw=true)

- coord_loss : loss of object's coordinates + loss of width, height about object bounding box

  ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/coord_loss.jpg?raw=true)

- noobject_loss : confidence loss for bounding box of each cell without an object

  ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/noobject_loss.jpg?raw=true)

  학습 초기에 값의 변화가 크게 있었지만, 이후 0으로 수렴하는것으로 보아 초기 loss의 순간적인 발산은 특정 data에 의해서 일시적으로 일어난 현상일 뿐임을 알 수 있다.

- object_loss : confidence loss for bounding box of each cell exist an object

  ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/object_loss.jpg?raw=true)

  전체적으로 loss의 튐 현상이 파동 형태로 나타나지만 결국 평균적인 loss는 점점 작은 값으로 수렴하는 모습을 보이고 있다.



**total_loss**

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/total_loss.jpg?raw=true)

total_loss는 값의 발산 없이 0의 값으로 수렴하고자 하는 모습을 볼 수 있다.



## Validation loss

- class_loss : 

  ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/total_validation_class_loss.jpg?raw=true)

- coord_loss : 

  ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/total_validation_coord_loss.jpg?raw=true)

- noobject_loss : 

  ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/total_validation_noobject_loss.jpg?raw=true)

- object_loss : 

  ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/total_validation_object_loss.jpg?raw=true)



**total loss**

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/total_validation_total_loss.jpg?raw=true)

전체적으로 Training의 각각의 loss보다 더욱 깔끔하게 loss가 0으로 수렴하는 모습을 볼 수 있다.

특히 object_loss의 파동형 발산은 보이지 않는 모습이다. 이를 통해 batch size를 1로 설정한 것 보다, batch size를 50으로 설정하고 training을 진행한다면 더욱 안정적인 학습이 이루어질 것이라는 기대를 가질 수 있다.



## Check model save, restore

model은 학습 과정에서 100×n 번째 step마다 save되도록 하고, 학습을 진행했다.

그리고 100, 200번째 step마다 학습을 중단시킨 후, 마지막에 저장된 model을 다시 가져와 학습하도록 해 보았다.

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/check_loss.jpg?raw=true)

위의 그래프는 total loss를 시각화 한 것이다. 보다시피 100번째, 200번째 step에서 학습을 시작 할 때 마다 loss는 random한 값으로 주어지는 weigh와 bias에 의해 큰 값으로 시작하지만, 이미 학습 된 model의 parameters에 의해 loss의 derivative가 높아 loss가 빠르게 낮아지는 것을 볼 수 있다. 





## Check learning rate decay

flags를 통해 initial learning rate는 0.0001로,  학습 과정에서 200번의 step마다 learning rate가 0.75씩 곱해져 적용되도록 해 놓았다.

```python
flags.DEFINE_float('init_learning_rate', default=0.0001, help='initial learning rate') # original paper : 0.001 (1epoch) -> 0.01 (75epoch) -> 0.001 (30epoch) -> 0.0001 (30epoch)

flags.DEFINE_float('lr_decay_rate', default=0.75, help='decay rate for the learning rate')

flags.DEFINE_integer('lr_decay_steps', default=200, help='number of steps after which the learning rate is decayed by decay rate') # 2000번 마다 init_learning_rate * lr_decay_rate 을 실행
```



그래프를 통해 learning rate의 decay가 잘 적용되었는지 확인해보자

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/learning_Rate.jpg?raw=true)

학습은 총 1k step까지 진행이 되었고, 매 200번째 step마다 0.75의 값이 learning rate에 곱해져 적용되었음을 확인할 수 있다.



## Evaluation with validation, test image

The right is the label image and the left is the predicted image by model

**validation image**

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/vaildation_image_0.jpg?raw=true)

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/vaildation_image_1.jpg?raw=true)



**test image**

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/0_result.png?raw=true)

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/1_result.png?raw=true)

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/2_result.png?raw=true)

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/9_result.png?raw=true)

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/6_result.png?raw=true)

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/11_result.png?raw=true)

큰 object들에 대해서는 detection이 잘 이루어지는것으로 보이지만, 작은 object에 대해서는 정확도가 낮은 것으로 판단된다.

또한 confidence가 가장 높은 boundingbox 1개만 표현하도록 했기 때문에 2개의 object에 대해서는 1개의 predictor만 표현되는 것을 확인할 수 있다.

