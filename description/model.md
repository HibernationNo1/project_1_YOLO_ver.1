# model.py

## YOLO의 기본 컨셉

image를 **S*S Grid Cell** 로 나누고, 각 cell 별로 **B** 개의 **Bounding Box**를 predict한다.

> 각 Bounding Box는 개당 5가지의 vector를 가지고 있다.



- object detection tesk에 의해 만들어진 Bounding Box 하나당 mapping되는 5가지 vector

  - x, y : grid cell 내의 object 중앙 x, y좌표 (normalization된 0~1 사이의 값)
  - width, height : 전체 image 대비 width, height 값 (normalization된 0~1 사이의 값)
  - confidence : image 내에 object가 있을 것이라고 확신하는 정도



- YOLO Model의 최종 output 개수
  $$
  output: S \times S \times(5*B + C)
  $$

  > **5** :  x, y coordinate, width, height, confidence 
  >
  > **B** : number of Bounding Box
  >
  > **C** : number of Class

  S = 7, B = 2, C = 20 일 경우 YOLO Model의 최종 output은 7 * 7 * 30



## Neural Network

### implement with keras.applications

Keras Model subclassing API와 Keras Functional API 방법으로 구현했으며, `tf.keras.applications`의 *Inception V3*(GoogLeNet응용 version) model을 가져온 후 GlobalAveragePooling2D과 Dense Layer을 추가해서 구현했다.  

[about applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications) 

| model or layer              | input shape                    | output shape                                          |             |                                                 |
| --------------------------- | ------------------------------ | ----------------------------------------------------- | ----------- | ----------------------------------------------- |
| InceptionV3.output          | (input_height, input_width, 3) | (None, 5, 5, 2048)                                    |             |                                                 |
| GlobalAveragePooling2D      |                                | (None, 2048)                                          |             |                                                 |
| Dense                       |                                | output                                                |             |                                                 |
| Pred_class Dense  Layer     |                                | units = cell_size * cell_size * num_classes           | → reshape → | [None, cell_size, cell_size, num_classse]       |
| Pred_confidence Dense Layer |                                | units = cell_size * cell_size * boxes_per_cell        | → reshape → | [None, cell_size, cell_size, boxes_per_cell]    |
| Pred_coordinate Dense Layer |                                | units = cell_size * cell_size *  (boxes_per_cell * 4) | → reshape → | [None, cell_size, cell_size, boxes_per_cell, 4] |
| **output**                  |                                | [Pred_class, Pred_confidence, Pred_coordinate]        |             |                                                 |



#### code

```python
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3

# Implementation using tf.keras.applications (https://www.tensorflow.org/api_docs/python/tf/keras/applications)
# & Keras Functional API (https://www.tensorflow.org/guide/keras/functional)
class YOLOv1(Model):
	def __init__(self, input_height, input_width, cell_size, boxes_per_cell, num_classes):
		super(YOLOv1, self).__init__()
		base_model = InceptionV3(include_top=False, weights='imagenet', 
								input_shape=(input_height, input_width, 3))
		# shape = (None, 5, 5, 2048)
		base_model.trainable = True
		x = base_model.output
    
		# Global Average Pooling
		x = GlobalAveragePooling2D()(x)  # shape = (None, 2048)
		x = Dense(cell_size * cell_size * (num_classes + (boxes_per_cell*5)), activation=None)(x)
		# flatten vector -> cell_size x cell_size x (num_classes + 5 * boxes_per_cell)
		pred_class = x[:,  : cell_size * cell_size * num_classes]
		pred_class = Dense(cell_size * cell_size * num_classes, activation=None)(pred_class)

		pred_confidence = x[:,  cell_size * cell_size * num_classes: (cell_size * cell_size * num_classes) + (cell_size * cell_size * boxes_per_cell)]
		pred_confidence = Dense(cell_size * cell_size * boxes_per_cell, activation=None)(pred_confidence)

		pred_coordinate = x[:, (cell_size * cell_size * num_classes) + (cell_size * cell_size * boxes_per_cell): ]
		pred_coordinate = Dense(cell_size * cell_size * (boxes_per_cell*4), activation=None)(pred_coordinate)		


		pred_class = tf.reshape(pred_class, 
				 [tf.shape(pred_class)[0], cell_size, cell_size, num_classes])

		pred_confidence = tf.reshape(pred_confidence, 
				 [tf.shape(pred_confidence)[0], cell_size, cell_size, boxes_per_cell])		

		pred_coordinate = tf.reshape(pred_coordinate,
				 [tf.shape(pred_coordinate)[0], cell_size, cell_size, boxes_per_cell, 4])

		output_list = [pred_class, pred_confidence, pred_coordinate]
	
		model = Model(inputs=base_model.input, outputs=output_list)
		self.model = model
		# print model structure
		self.model.summary()

	def call(self, x):
		return self.model(x)
```





### implement without keras.applications

Keras Model subclassing API로 구현. Class 안에서는 Keras Functional API 방식으로 YOLO의 기본 형태와 같은 구성으로 layer을 쌓았다.

![img](https://i0.wp.com/thebinarynotes.com/wp-content/uploads/2020/04/Yolo-Architecture.png?fit=678%2C285&ssl=1)





| input image                 | 448 × 448 × 3       |                                                              |             |                                                              |
| --------------------------- | ------------------- | ------------------------------------------------------------ | ----------- | ------------------------------------------------------------ |
| back born network           | Darknet             |                                                              |             |                                                              |
| Convolution                 | kernel size = 7     | num of kernel = 64                                           | strides = 2 | padding = [kernel size/2]                                    |
| Maxpooling                  | kernel size = 2     |                                                              | strides = 2 |                                                              |
| **feature map1**            | **112 × 112 × 192** |                                                              |             |                                                              |
| Convolution                 | kernel size = 3     | num of kernel = 192                                          | strides = 1 | padding = [kernel size]                                      |
| **feature map2**            | **112 × 112 × 64**  |                                                              |             |                                                              |
|                             | **112 × 112 × 256** | **staking feature map 1, 2**                                 |             |                                                              |
| Maxpooling                  | kernel size = 2     |                                                              | strides = 2 |                                                              |
|                             | **56 × 56 × 256**   |                                                              |             |                                                              |
| Convolution                 | kernel size = 1     | num of kernel = 128                                          | strides = 1 | padding = [kernel size]                                      |
| Convolution                 | kernel size = 3     | num of kernel = 256                                          | strides = 1 | padding = [kernel size/2]                                    |
| Convolution                 | kernel size = 1     | num of kernel = 256                                          | strides = 1 | padding = [kernel size]                                      |
| Convolution                 | kernel size = 3     | num of kernel = 512                                          | strides = 1 | padding = [kernel size/2]                                    |
| Maxpooling                  | kernel size = 2     |                                                              | strides = 2 |                                                              |
|                             | **28 × 28 × 512**   |                                                              |             |                                                              |
| Conv 1                      | kernel size = 1     | num of kernel = 256                                          | strides = 1 | padding = [kernel size]                                      |
| Conv 2                      | kernel size = 3     | num of kernel = 512                                          | strides = 1 | padding = [kernel size/2]                                    |
|                             |                     | **4 times iteration conv 1, 2**                              |             |                                                              |
| Convolution                 | kernel size = 1     | num of kernel = 512                                          | strides = 1 | padding = [kernel size]                                      |
| Convolution                 | kernel size = 3     | num of kernel = 1024                                         | strides = 1 | padding = [kernel size/2]                                    |
| Maxpooling                  | kernel size = 2     |                                                              | strides = 2 |                                                              |
|                             | **14 × 14 × 1024**  |                                                              |             |                                                              |
| Conv 3                      | kernel size = 1     | num of kernel = 512                                          | strides = 1 | padding = [kernel size]                                      |
| Conv 4                      | kernel size = 3     | num of kernel = 1024                                         | strides = 1 | padding = [kernel size/2]                                    |
|                             |                     | **2 times iteration conv 3, 4**                              |             |                                                              |
| Convolution                 | kernel size = 3     | num of kernel = 1024                                         | strides = 1 | padding = [kernel size/2]                                    |
| Convolution                 | kernel size = 3     | num of kernel = 1024                                         | strides = 2 | padding = [kernel size/2]                                    |
|                             | **7 × 7 × 1024**    |                                                              |             |                                                              |
| Convolution                 | kernel size = 3     | num of kernel = 1024                                         | strides = 1 | padding = [1 + kernel size/2]                                |
| Convolution                 | kernel size = 3     | num of kernel = 1024                                         | strides = 1 | padding = [1 + kernel size/2]                                |
|                             | **7 × 7 × 1024**    |                                                              |             |                                                              |
| GlobalAveragePooling        |                     |                                                              |             |                                                              |
|                             | **1 × 1024**        |                                                              |             |                                                              |
| Dense Layer                 |                     | units = cell_size * cell_size * (num_classes + (boxes_per_cell*5) | → divide →  | Pred_class Dense  Layer,<br />Pred_confidence Dense Layer<br />Pred_coordinate Dense Layer |
|                             |                     |                                                              |             |                                                              |
| Pred_class Dense  Layer     |                     | units = cell_size * cell_size * num_classes                  | → reshape → | [None, cell_size, cell_size, num_classse]                    |
| Pred_confidence Dense Layer |                     | units = cell_size * cell_size * boxes_per_cell               | → reshape → | [None, cell_size, cell_size, boxes_per_cell]                 |
| Pred_coordinate Dense Layer |                     | units = cell_size * cell_size *  (boxes_per_cell * 4)        | → reshape → | [None, cell_size, cell_size, boxes_per_cell, 4]              |
| **output**                  |                     | [Pred_class, Pred_confidence, Pred_coordinate]               |             |                                                              |



#### code

```python
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import ZeroPadding2D, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.regularizers import L1, L2

from functools import reduce


# Implementation using tf.keras.applications (https://www.tensorflow.org/api_docs/python/tf/keras/applications)
# & Keras Functional API (https://www.tensorflow.org/guide/keras/functional)
class YOLOv1(Model):
	def __init__(self, input_height, input_width, cell_size, boxes_per_cell, num_classes):
		super(YOLOv1, self).__init__()
		input_tensor = Input(shape = (input_height, input_width, 3))
		# shape = [None, 448, 448, 3]

		#x = Conv2D(filters = 64, kernel_size = 7, strides = 2, padding = 'valid')(input_tensor)
		#x = BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001)(x)
		#x = Dropout(0.2)(x)
		#x = LeakyReLU(alpha = 0.1)(x)
		#x = MaxPooling2D(pool_size = 2, strides = 2)(x)
		x = self.compose([self.maxpooling(2, 2), self.padding(int(7/2)), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(64, 7, 2)])(input_tensor)
		# shape = [None, 112, 112, 64]

		x = self.compose([self.maxpooling(2, 2), self.padding(3), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(192, 3, 1)])(x)
		# shape = [None, 56, 56, 192]

		x = self.compose([self.padding(1), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(128, 1, 1)])(x)
		x = self.compose([self.padding(int(3/2)), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(256, 3, 1)])(x)
		x = self.compose([self.padding(1), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(256, 1, 1)])(x)
		x = self.compose([self.maxpooling(2, 2), self.padding(int(7/2)), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(512, 3, 1)])(x)
		# shape = [None, 28, 28, 512]

		x = self.compose([self.padding(1), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(256, 1, 1)])(x)
		x = self.compose([self.padding(int(3/2)), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(512, 3, 1)])(x)
		x = self.compose([self.padding(1), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(256, 1, 1)])(x)
		x = self.compose([self.padding(int(3/2)), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(512, 3, 1)])(x)
		x = self.compose([self.padding(1), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(256, 1, 1)])(x)
		x = self.compose([self.padding(int(3/2)), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(512, 3, 1)])(x)
		x = self.compose([self.padding(1), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(256, 1, 1)])(x)
		x = self.compose([self.padding(int(3/2)), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(512, 3, 1)])(x)
		x = self.compose([self.padding(1), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(512, 1, 1)])(x)
		x = self.compose([self.maxpooling(2, 2), self.padding(int(3/2)), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(1024, 1, 1)])(x)
		# shape = [None, 14, 14, 1024]

		x = self.compose([self.padding(1), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(512, 1, 1)])(x)
		x = self.compose([self.padding(int(3/2)), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(1024, 3, 1)])(x)
		x = self.compose([self.padding(1), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(512, 1, 1)])(x)
		x = self.compose([self.padding(int(3/2)), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(1024, 3, 1)])(x)
		x = self.compose([self.padding(int(3/2)), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(1024, 3, 1)])(x)
		x = self.compose([self.padding(int(3/2)), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(1024, 3, 2)])(x)
		# shape = [None, 7, 7, 1024]

		x = self.compose([self.padding(2), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(1024, 3, 1)])(x)
		x = self.compose([self.padding(2), LeakyReLU(alpha = 0.1), Dropout(0.2), 
						BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001), self.conv(1024, 3, 1)])(x)
		# shape = [None, 7, 7, 1024]

		# Global Average Pooling
		x = GlobalAveragePooling2D()(x)  # shape = (None, 1024)
		x = Dense(cell_size * cell_size * (num_classes + (boxes_per_cell*5)), activation=None)(x)
		# flatten vector -> cell_size x cell_size x (num_classes + 5 * boxes_per_cell)
		pred_class = x[:,  : cell_size * cell_size * num_classes]
		pred_class = Dense(cell_size * cell_size * num_classes, activation=None, kernel_regularizer = L2(l=0.01))(pred_class)

		pred_confidence = x[:,  cell_size * cell_size * num_classes: (cell_size * cell_size * num_classes) + (cell_size * cell_size * boxes_per_cell)]
		pred_confidence = Dense(cell_size * cell_size * boxes_per_cell, activation=None, kernel_regularizer = L2(l=0.03))(pred_confidence)

		pred_coordinate = x[:, (cell_size * cell_size * num_classes) + (cell_size * cell_size * boxes_per_cell): ]
		pred_coordinate = Dense(cell_size * cell_size * (boxes_per_cell*4), activation=None, kernel_regularizer = L1(l=0.1))(pred_coordinate)		


		pred_class = tf.reshape(pred_class, 
				 [tf.shape(pred_class)[0], cell_size, cell_size, num_classes])

		pred_confidence = tf.reshape(pred_confidence, 
				 [tf.shape(pred_confidence)[0], cell_size, cell_size, boxes_per_cell])		

		pred_coordinate = tf.reshape(pred_coordinate,
				 [tf.shape(pred_coordinate)[0], cell_size, cell_size, boxes_per_cell, 4])

		output_list = [pred_class, pred_confidence, pred_coordinate]
	
		model = Model(inputs=input_tensor, outputs=output_list)
		self.model = model
		# print model structure
		self.model.summary()

	def conv(self, num_filters, k_size, strd):
		return Conv2D(filters = num_filters, kernel_size = k_size, strides = strd, padding = 'valid')

	def maxpooling(self, p_size , strd):
		return MaxPooling2D(pool_size = p_size, strides = strd)

	def padding(self, p_size):
		return ZeroPadding2D(padding= p_size)

	def compose(self, functions):
		return reduce(lambda f, g: lambda x: f(g(x)), functions)

	def call(self, x):
		return self.model(x)
```

- overfitting을 예방하기 위해 regularization방법으로 conv layer 직후 dropout layer를 사용했으며, dense layer에는 L1 Regularization, L2 Regularization을 적용했다.

- internal Convariate Shift 문제를 해결하기 위해 Batch Normalization layer를 사용했다.

- ```python
     def batch_normal(self):
      	return BatchNormalization(axis = -1, momentum=0.99, epsilon=0.001)   
  ```

  해당 method를 정의 후 compose 안에서 BatchNormalization 대체로 호출하면 
   `batch_normal() takes 1 positional argument but 2 were given` 이라는 이슈 발생. 원인 불명.

  같은 반환으로 method 없이 직접 호출하면 argument == 1로 이상 없이 동작한다.

  수정사항



#### compare inception.V3 and without inception.V3

input width, height를 224에서 448로 수정 후 학습을 진행시켰다.

inception.V3을 포함한 model은 150step까지, inception.V3을 포함하지 않은 model은 100step까지 학습한 것의 경과를 비교해보았다.

- **class loss**
  - inception.V3
  
    ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/calss_loss.png?raw=true)
  
  - without inception.V3
  
    ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/noap_calss_loss.png?raw=true)
- **coordinate loss**
  - inception.V3
  
    ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/coord_loss.png?raw=true)
  
  - without inception.V3
  
    ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/noap_coord_loss.png?raw=true)
- **object loss**
  - inception.V3
  
    ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/object_loss.png?raw=true)
  
  - without inception.V3
  
    ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/noap_object_loss.png?raw=true)
- **noobject loss**
  - inception.V3
  
    ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/noobject_loss.png?raw=true)
  
  - without inception.V3
  
    ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/noap_noobject_loss.png?raw=true)
- **total loss**
  
  - inception.V3
  
    ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/total_loss.png?raw=true)
  
  - without inception.V3
  
    ![](https://github.com/HibernationNo1/project_YOLO_ver.1/blob/master/description/image/noap_total_loss.png?raw=true)





## display process of bounding box

1. #### **YOLO Model에 image를 input**

   image를 **S*S Grid Cell** 로 나누고, 각 cell 별로 **B** 개의 **Bounding Box**를 predict한다.

   

2. #### **class specific confidence score**

   output image의 grid cell 마다 각 class에 대한 probability를 각 bounding box의 confidence와 곱한 vector를 연산한다.



3. #### **non-*maximum* suppression**

   class specific confidence score 로 인해 나온 다수의 bounding box 중 가장 대표성을 띄는 bounding box만 표현한다.

   ##### non-*maximum* suppression 동작 과정

   1. confidence 가 0.6 이하인 bounding box를 제거한다
   2. class별로 confidence가 가장 높은 bounding box가 앞으로 오도록 전체 bounding box를 내림차순 정렬한다.
   3. 가장 confidence가 높은 bounding box와 나머지 bounding bax를 비교해서 2개의 bounding box의 IoU 가 0.5보다 크다면 confidence가 가장 높은 bounding box를 제거한다
   4. 제거되지 않는 bounding box 중에서 confidence가 가장 높은 bounding box와 나머지 bounding box간에 3번 과정을 반복한다.
   5. 3~ 4번 과정을 전체 bounding box에 대해서 진행한다.
   6. 2~5번 과정을 전체 class에 대해서 진행한다.

   > 3~4번 과정이 non-*maximum* suppression algorithm이다



**exmple : S = 7, B = 2, C = 20 일 경우**
![](https://curt-park.github.io/images/yolo/DeepSystems-NetworkArchitecture.JPG)





