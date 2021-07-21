# READ ME

이 프로젝트는 YOLO v.1의 논문을 읽고 구현해보는 [인프런](https://www.inflearn.com/)의 강의를 참고하여 진행했습니다.

paper Reference : https://www.inflearn.com/course/%EC%9A%9C%EB%A1%9C-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EB%85%BC%EB%AC%B8%EA%B5%AC%ED%98%84/dashboard



**강의 교육 내용**

논문을 통해 YOLO의 개념과 loss function을 파악 후 구현 후 model training 및 evaluation을 진행합니다.



**Add new function**

강의 내용에 더해 개인적으로 추가하고 싶은 기능 몇 가지를 추가했습니다.

- **CONTINUE_LEARNING**

  training이 unintentionally하게 중단되었거나, data의 update로 인해 model을 새롭게 training해야 하는 경우를 위해 continue 여부에 대한 flag를 추가했습니다.

  해당 flag의 True, False 여부에 따라 directory생성, 삭제, load saved model 등의 동작이 이루어집니다.

  `CONTINUE_LEARNING = False` : 이전에 했던 training을 다시 시작하는 경우

  `CONTINUE_LEARNING = True `: 이전에 했던 training의 step에 이어서 진행 할 경우

  

  

**Performance improvement**

performation 개선을 위해 기존 code에 몇 가지 수정사항이 적용되었습니다.

- 





비교 분석



**Contents**

- [Code](#Code)
  - [Utilities](#Utilities)
  - [model.py](#model.py)
  - [train.py](#train.py)
  - [evaluate.py](#evaluate.py)
- [Result](#Result)
- [Getting Started](#'Getting Started')
- [version](#version)



## Code

### Utilities

- [dataset.py](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/dataset.md) : Load data **PASCAL VOC 2007, 2012** and perform pre-processing.
- [utils.py](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/utils.md) : Contain some utilities function for model training.
- [loss function.py](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/loss.md#yolo_loss) : Defines the loss function used in the YOLO model.





### model.py

[model.py](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/model.md) : Implement the YOLO model



### train.py

[train.py](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/train.md) : 

- Create instance of model class and do gradient descent through `for-loop` for parameter updata
- When the iteration reaches a certain number of times, a validation is performed.
- All training logs and validation logs, label and prediction image comparisons are saved in the tensorboard.



### evaluate.py

[evaluate.py](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/evaluate.md) : 

- The test is performed using the parameters of the trained model.

- prediction result about test image and test result are saved to directory named 'test result'  as `png` file.



## Result

**detail analysis** : [here](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/Analysis.md)

Training will continue as 10 epochs progress, Validation was performed every 50 steps.

I use tensorboard to show valuse

**train total_loss**

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/total_loss.jpg?raw=true)



**validation total_loss**

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/total_validation_total_loss.jpg?raw=true)



**evaluation with test data**

The right is the label image and the left is the predicted image by model

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/5_result.png?raw=true)

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/9_result.png?raw=true)

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/10_result.png?raw=true)

![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/image/11_result.png?raw=true)



It seems that detection works well for large objects, but have low accuracy for small objects.

It detect only one object if two or more objects exist in image because it made to represent only the one bounding box with the highest confidence.



## Getting Started

#### training

```
$ code\train.py
```

> `main(_)` function is included in code



#### evaluation

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

