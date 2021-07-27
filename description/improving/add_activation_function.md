학습에 사용되는 units를 목적에 맞게 분류 후 적절한 loss를 계산하기 위한 activation function을 적용했다.



- yolo의 loss function의 요소 중 class loss는 multy class에 대한 각각의 probability를 예측하기 위해 **CategoricalCrossentropy**를 적용해 계산한다.

  이를 위해서는 해당 prediction value는 probability를 표현하기 위해 softmax function이 적용되어야 하지만(softmax function는 베르누이 분포를 상정하기 때문), code에서는 softmax function이 없음을 확인했다.

  

- yolo의 loss function의 요소 중 confidence loss는 두 개의 Bbox에 대한 각각의 probability를 예측하기 위해 **BinaryCrossentropy**를 적용해 계산한다.

  이를 위해서는 해당 prediction value는 probability를 표현하기 위해 sigmoid function이 적용되어야 하지만,  code에서는 sigmoid function이 없음을 확인했다.

  

### todo list

- flatten shape의 units을 각 용도에 맞게 분류 휴 cell_size × cell_size 의 shape으로 reshape

- classification에 사용할 units에 softmax function 적용
- confidence predict에 사용할 units에 sigmoid function 적용



### improve

**변경 전 model.py**

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
		output = Dense(cell_size * cell_size * (num_classes + (boxes_per_cell*5)), activation=None)(x)
		model = Model(inputs=base_model.input, outputs=output)
		self.model = model
		# print model structure
		self.model.summary()

	def call(self, x):
		return self.model(x)
```





**변경 후 model.py**

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
		pred_class = x[:,  : cell_size * cell_size * num_classes]
		pred_confidence = x[:,  cell_size * cell_size * num_classes: (cell_size * cell_size * num_classes) + (cell_size * cell_size * boxes_per_cell)]
		pred_coordinate = x[:, (cell_size * cell_size * num_classes) + (cell_size * cell_size * boxes_per_cell): ]
		
		pred_class = tf.reshape(pred_class, 
				 [tf.shape(pred_class)[0], cell_size, cell_size, num_classes])
		pred_class = tf.nn.softmax(pred_class) 
		# tf.reduce_sum(pred_class) == cell_size * cell_size

		pred_confidence = tf.reshape(pred_confidence, 
				 [tf.shape(pred_confidence)[0], cell_size, cell_size, boxes_per_cell])
		pred_confidence = tf.sigmoid(pred_confidence)
		

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

