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
		pred_confidence = x[:,  cell_size * cell_size * num_classes: (cell_size * cell_size * num_classes) + (cell_size * cell_size * boxes_per_cell)]
		pred_coordinate = x[:, (cell_size * cell_size * num_classes) + (cell_size * cell_size * boxes_per_cell): ]
		
		pred_class = tf.reshape(pred_class, 
				 [tf.shape(pred_class)[0], cell_size, cell_size, num_classes])
		pred_class = tf.nn.softmax(pred_class) 
		# tf.reduce_sum(pred_class) == cell_size * cell_size

		pred_confidence = tf.reshape(pred_confidence, 
				 [tf.shape(pred_confidence)[0], cell_size, cell_size, boxes_per_cell])
		pred_confidence = tf.nn.sigmoid(pred_confidence)		

		pred_coordinate = tf.reshape(pred_coordinate,
				 [tf.shape(pred_coordinate)[0], cell_size, cell_size, boxes_per_cell, 4])

		output_list = [pred_class, pred_confidence, pred_coordinate]
	
		model = Model(inputs=base_model.input, outputs=output_list)
		self.model = model
		# print model structure
		self.model.summary()

	def call(self, x):
		return self.model(x)