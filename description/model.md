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



### Neural Network

Keras Model subclassing API로 구현. Class 안에서는 Keras Functional API 방식으로 구글의 *Inception V3* model을 가져온 후 GlobalAveragePooling2D과 Dense Layer을 추가해서 구현했다.  

| model or layer         | input | output |
| ---------------------- | ----- | ------ |
| InceptionV3.output     | x0    | x1     |
| GlobalAveragePooling2D | x1    | x2     |
| Dense                  | x2    | output |





## Training process

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





## Code

Implementated using tf.keras.applications 

[about applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications) 

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

- Keras Functional API (https://www.tensorflow.org/guide/keras/functional)

| model or layer         | input | output          | output shape                                                 |
| ---------------------- | ----- | --------------- | ------------------------------------------------------------ |
| InceptionV3.output     | x     | x               | (None, 5, 5, 2048)                                           |
| GlobalAveragePooling2D | x     | x               | (None, 2048)                                                 |
| Dense                  | x     | x               | (None, (cell_size x cell_size x (num_classes + 5 * boxes_per_cell)) |
| pred_class Dense       | x     | pred_class      | (None, cell_size x cell_size x num_classes)                  |
| pred_confidence Dense  | x     | pred_confidence | (None, cell_size x cell_size x boxes_per_cell)               |
| pred_coordinate Dense  | x     | pred_coordinate | (None, cell_size x cell_size x (4 * boxes_per_cell)          |







