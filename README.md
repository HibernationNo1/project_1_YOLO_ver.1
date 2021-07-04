# READ ME

This project is a YOLO v.1 implementation project comleted by referring to online cource '[Inflearn](https://www.inflearn.com/)', which thesis implementation cource. 

The code structure and some functions have been changed for personal convenience.



Reference : https://www.inflearn.com/course/%EC%9A%9C%EB%A1%9C-%EB%94%A5%EB%9F%AC%EB%8B%9D-%EB%85%BC%EB%AC%B8%EA%B5%AC%ED%98%84/dashboard



**Contents**

- [Code](#Code)
  - [Utilities](#Utilities)
  - [model.py](#model.py)
  - [train.py](#train.py)
  - [evaluate.py](#evaluate.py)
- [Result](#Result)
- [Getting Started](#Getting Started)
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

