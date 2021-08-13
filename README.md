# READ ME

이 프로젝트는 YOLO v.1의 논문을 읽고 구현해보는 [인프런](https://www.inflearn.com/)의 강의를 참고하여 진행했습니다.

[paper Reference](https://www.inflearn.com/course/%EC%9A%9C%EB%A1%9C-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EB%85%BC%EB%AC%B8%EA%B5%AC%ED%98%84/dashboard)

**강의 교육 내용**

해당 강의는 YOLO V.1의 논문을 읽고 구현하는 강의입니다. 구현하는 code는 20개의 class 중 1개만을 학습하고 test하는 과정으로 이루어져 있으며, 제공되는 code는 가장 기초적인 부분만 구현되어 있습니다.



**project를 진행한 목적**

해당 강의에서 제공되는 code는 YOLO model의 initial version이며, single class를 학습하고 여러 계산 과정에 오류가 있는 등 날것 그대로의 형태를 가지고 있습니다. 저는 이러한 code를 직접 개선하고 발전시켜나가며 스스로의 역량을 키우고자 프로젝트를 진행하게 되었습니다. 개선 사항에 대해서는 [improvement list](#improvement-list) 에 기록하였습니다.



**Contents**

- [improvement list](#improvement-list)

- [Code](#Code)
  - [Utilities](#Utilities)
  - [model.py](#model.py)
  - [train.py](#train.py)
  - [evaluate.py](#evaluate.py)
- [Result](#Result)
- [Getting Started](#Getting-Started)
- [version](#version)





---





## improvement list

- **CONTINUE_LEARNING**

  training이 unintentionally하게 중단되었거나, data의 update로 인해 model을 새롭게 training해야 하는 경우를 위해 continue 여부에 대한 flag를 추가했습니다.

  해당 flag의 True, False값에 따라 directory생성, 삭제, load saved model 등의 동작이 이루어집니다.

  `CONTINUE_LEARNING = False` : 이전에 했던 training을 다시 시작하는 경우

  `CONTINUE_LEARNING = True `: 이전에 했던 training의 step에 이어서 진행 할 경우

  [detail](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/improvements/CONTINUE_LEARNING.md)

- **multi_object_detection**

  기존의 code는 dataset에서 1개의 class만 extraction하고, test result로 1개의 Bbox만을 표현했습니다.

  저는 multi class에 대한 학습과 test를 위해 dataset에서 여러개의 class를 extraction하는 것으로 변경하였고, test result 또한 특정 조건(confidence score)을 만족한 여러개의 Bbox를 표현하도록 했습니다.

  [detail](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/improvements/multi_object_detection.md)

- **confidence_score**

  test result로 Bbox를 표현할 때 조건 없이 class probability가 가장 높은 Bbox를 표현하는 것을 확인했습니다. 이는 기준으로 삼는 confidence score의 값은 단순히 크지만 전혀 엉뚱한 위치의 object를 가리키는 Bbox를 표현하게 되는 문제점을 가지고 있습니다. 

  이를 방지하기 위해 iou의 개념을 통해 실제 object가 존재하는 위치에 가까울수록 donfidence score가 높을 수 있도록 confidence score = (intersection_of_union) * (predited class probability) 으로 표현했습니다.

  또한 multi object detection을 위해 각 object마다 [cell_size, cell_size, box_per_cell] shape의 iou를 계산하도록 했습니다.

  [detail](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/improvements/confidence_score.md)

- **remove_irrelevant_label**

  기존의 code는 dataset에서 data를 추려낼 때 target class object가 하나라도 포함 된 data는 모두 추려냅니다. 이 과정에서 target class가 아닌 class의 object도 포함된 data가 있는데, 이러한 object는 학습 목표과 관련이 없는 label이기 때문에 loss를 높히는 원인이 됨을 확인했습니다.

  > **ex)** cat, dog만을 학습시키고자 할 때 image에 cat, human, cow 등 target이 아닌 label이 포함되어 있으면 이에 대한 loss가 계산되어 학습에 방해가 된다.

  이를 해결하기 위해 label에 포함된 class중 target class외의 모든 class에 대한 value를 0으로 만드는 function을 정의했습니다.

  [detail](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/improvements/remove_irrelevant_label.md)

- **improve_total_loss**  

  기존 code는 각각의 loss를 계산하는 과정에서 모두 MSE를 사용했습니다. 

  이 과정에서 학습이 제대로 되지 않는 과정을 확인, 여러가지 사항을 수정했습니다. 

  - 기존 code는 confidence loss의 label value를 interception of union으로 사용하는데, 이때

    initial prediction value는 학습이 되지 않은 값이기 때문에 0에 가까운 값을 반환하게 됩니다. 그렇기 때문에 학습의 처음부터 잘못된 loss값이 사용되는 것을 확인했습니다.

    또한 confidence loss는 object의 존재 여부에 대한 학습이기 때문에  MSE보다 cross entropy를 사용함이 더욱 적절하다고 판단하여  `tf.nn.sigmoid_cross_entropy_with_logits()`를 적용했으며, 이를 위해서 label값을 구성하는 방법을 수정했습니다.

  - 기존 code는 class loss의 label value에 적용된 one-hot encoding에서 label에 대한 number가 제대로 된 전달이 이루어지지 않아 0.0만을 반환하는 문제점을 확인했습니다. 이는 학습이 진행될수록 class loss는 0에 가까운 값으로 수렴하기 때문에 각 label값에 알맞게 one-hot encoding이 적용된 값이 할당될 수 있도록 function을 추가했습니다.

    또한 class loss는 예측한 특정 class에 대한 probability를 표현하기 때문에 사용되는 loss functiond은 MSE가 아닌 `tf.nn.softmax_cross_entropy_with_logits`으로 적용했습니다.

  [detail](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/improvements/improve_loss_function.md)

- **performance_evaluation**

  validation, test result에 대한 performance evaluation function을 추가했습니다.

  성능 평가는 세 가지 경우를 고려했습니다.

  - **average_detection_rate**

    result에 대한 average object detection rate입니다.

    특정 조건을 만족하는 Bbox가 존재하는 경우에 대한 비율을 계산합니다. 

    Performance Evaluation Index 중 **Recall**의 방법을 따랐습니다.
  
    
    $$
    Detection\ Rate = \frac{Num\ Detected\ Object}{Num\ Label\ Object} * 100%
    $$
  
    $$
    Average\ Detection\ Rate = \frac{Sum \ Detection\ Rate }{Num\ Test\ Image}
    $$

    ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/1.png?raw=true)

  - **perfect_detection_accuracy**
  
    object detection이 이루어진 result중 완벽한 object detection이 이루어진 비율입니다.
  
    
    $$
    Perfect\ Detection\ Accuracy = \frac{Num\ Perfect\ Detection }{Num\ Test\ Image}
    $$
    ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/2.png?raw=true)
  
  
    > label object가 1개일 때 2개 이상을 감지하면 over detection
    >
    > label object가 2개일 때 1개만을 감지하면 low detection
    >
    > label object가 n개일 때 n를 감지하면 perfect detection
  
    위의 detection_rate == 100% 인 경우 perfect detection인 것으로 결정했습니다.
  
  - **classification_accuracy**
  
    result에 대한 대한 정확한 classification이 이루어진 비율입니다.
  
    perfect detection이라는 전제 조건에서 성공적인 classification가 이루어졌는지 확인합니다. (즉, perfect detection인 경우가 아니면 success classification 확인 과정을 수행하지 않았습니다.)
    
    
    $$
    Classification Accuracy = \frac{Num\ Correct\ Answers\ Class }{Num\ Test\ Image}
    $$
    
    ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/3.png?raw=true)
    
    *success classification 확인 과정*
    
    1. label과 prediction의 object list를 x좌표 기준으로 올림차순 정렬을 수행한다.
    2. x좌표가 낮은 object부터 x좌표가 높은 object 순으로 label과 prediction의 class name이 동일한지 확인한다.
    3. 2번의 조건이 만족하면, label과 prediction의 object list를 y좌표 기준으로 올림차순 정렬을 수행한다.
    4. y좌표가 낮은 object부터 y좌표가 높은 object 순으로 label과 prediction의 class name이 동일한지 확인한다.
    5. 1, 2, 3, 4번의 동작에서 모든 조건에 부합한 경우라면, success classification인 것으로 간주한다.
  
  [detail](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/improvements/performance_evaluation.md)





## Code

### Utilities

- [dataset.py](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/dataset.md) : Make directory, load data **PASCAL VOC 2007, 2012** and perform pre-processing.
- [utils.py](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/utils.md) : Contain some utilities function for model training.
- [loss function.py](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/loss.md#yolo_loss) : Defines the loss function used in the YOLO model.



### model.py

[model.py](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/model.md) : Implementing the YOLO model



### train.py

[train.py](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/train.md) : 

- Creating instance of model class and do gradient descent through `for-loop` for parameter updata
- When the iteration reaches a certain number of times, a validation is performed.
- All training logs and validation logs, label and prediction image comparisons are saved in the tensorboard.



### evaluate.py

[evaluate.py](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/evaluate.md) : 

- The test is performed using the parameters of the trained model.

- prediction result about test image and test result are saved to directory named 'test result'  as `png` file.



## Result

**detail analysis** : [here](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/Analysis.md)

Training will continue as 1000step progress, Validation was performed every 100 steps.

I use tensorboard to show valuse

**train total_loss**

I have restarted when 1000 step for check CONTINUE_LEARNING flags

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/main_total_loss.png?raw=true)



**validation total_loss**

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/validation_total_loss.png?raw=true)



**evaluation with test data**

The right is the label image and the left is the predicted image by model

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/0_result.png?raw=true)

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/409_result.png?raw=true)

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/431_result.png?raw=true)

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/6_result.png?raw=true)







## Getting Started

#### training

```
$ code\train.py
```

> `main(_)` function is included in code



#### evaluate

```
$ code\evaluate.py
```

> `main(_)` function is included in code



## version

| name                | version |
| ------------------- | ------- |
| python              | 3.8.5   |
|                     |         |
| **package name**    |         |
| numpy               | 1.19.5  |
| tensorflow          | 2.5.0   |
| tensorflow_datasets | 4.3.0   |
| cv2                 | 4,5,2   |
| sys                 | 3.8.8   |

