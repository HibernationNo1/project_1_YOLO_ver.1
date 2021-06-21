# model

Implementation using tf.keras.applications 

[about applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications) 

```python
import tensorflow as tf



class YOLOv1(tf.keras.Model):
    def __init__(self, input_height, input_width, cell_size, boxes_per_cell, num_classes):
        super(YOLOv1, self).__init__()
        base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(input_height, input_width, 3))
        # Bring GoogLeNet ver.3, only feature extractor part
        base_model.trainable = True
        # include parameters of GoogLeNet ver.3 for training
        # To get better performance
        x1 = base_model.output
        # Using keras Functional API
        # x == feature map

        # Global Average Pooling
        x2 = tf.keras.layers.GlobalAveragePooling2D()(x1)
        output = tf.keras.layers.Dense(cell_size * cell_size * (num_classes + (boxes_per_cell*5)), activation=None)(x2)
        # resizing shape likes (7, 7, 30) by Dense layer  
        model = tf.keras.Model(inputs=base_model.input, outputs=output)
        self.model = model
        # print model structure
        self.model.summary()

    def call(self, x0):
        return self.model(x0)
```

- Keras Functional API (https://www.tensorflow.org/guide/keras/functional)

| model or layer         | input | output |
| ---------------------- | ----- | ------ |
| InceptionV3.output     | x0    | x1     |
| GlobalAveragePooling2D | x1    | x2     |
| Dense                  | x2    | output |

