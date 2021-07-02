# READ ME

## Abstract

this project is a YOLO v.1 implementation project comleted by referring to online cource '[Inflearn](https://www.inflearn.com/)', which thesis implementation cource. 

The code structure and some functions have been changed for personal convenience.



Reference : 



## Code



### Utilities

- [dataset.py](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/dataset.md) : Load data **PASCAL VOC 2007, 2012** and perform pre-processing.
- [utils.py](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/utils.md) : Contain some utilities function for model training.
- [loss function.py](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/loss.md#yolo_loss) : Defines the loss function used in the YOLO model.





### model.py

[model.py](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/model.md) : Implement the YOLO model



### train.py

[train.py](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/train.md) : `main(_)` function is included in code

- Create instance of model class and do gradient descent through `for-loop` for parameter updata
- When the iteration reaches a certain number of times, a validation is performed.
- All training logs and validation logs, label and prediction image comparisons are saved in the tensorboard.



### evaluate.py

[evaluate.py](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/evaluate.md) : 

- The test is performed using the parameters of the trained model.

- prediction result about test image and test result are saved to directory named 'test result'  as `png` file.



## Result





## Conclusion





### Getting Started

#### training

```
$ code\train.py
```



#### evaluation

```
$ code\evaluate.py
```



### version

| name                | version |
| ------------------- | ------- |
| python              | 3.8.5   |
|                     |         |
| **package name**    |         |
| numpy               |         |
| tensorflow          |         |
| tensorflow_datasets |         |
| cv2                 |         |
| colorsys            |         |
| random              |         |
| sys                 |         |
| os                  |         |
| shutil              |         |

