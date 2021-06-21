import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layer import Dense
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D

# Implementation using tf.keras.applications (https://www.tensorflow.org/api_docs/python/tf/keras/applications)
# & Keras Functional API (https://www.tensorflow.org/guide/keras/functional)
class YOLOv1(Model):
    def __init__(self, input_height, input_width, cell_size, boxes_per_cell, num_classes):
        super(YOLOv1, self).__init__()
        base_model = InceptionV3(include_top=False, weights='imagenet', 
                                input_shape=(input_height, input_width, 3))
        base_model.trainable = True
        x = base_model.output

        # Global Average Pooling
        x = GlobalAveragePooling2D()(x)
        output = Dense(cell_size * cell_size * (num_classes + (boxes_per_cell*5)), activation=None)(x)
        model = Model(inputs=base_model.input, outputs=output)
        self.model = model
        # print model structure
        self.model.summary()

    def call(self, x):
        return self.model(x)