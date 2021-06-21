# utils

import

```python
import cv2
import numpy as np
import tensorflow as tf
import colorsys
from operator import itemgetter
```



#### draw_bounding_box_and_label_info

```python
def draw_bounding_box_and_label_info(frame, x_min, y_min, x_max, y_max, label, confidence, color):
  draw_bounding_box(frame, x_min, y_min, x_max, y_max, color)
  draw_label_info(frame, x_min, y_min, label, confidence, color)
```



##### draw_bounding_box

```python
def draw_bounding_box(frame, x_min, y_min, x_max, y_max, color):
      cv2.rectangle(
        frame,
        (x_min, y_min),
        (x_max, y_max),
        color, 3)

```

- using OpenCV

  `cv2.rectangle(img, pt1, pt2, color, thickness = None, lineType = None, shift = None)`

  `pt1, pt2` : 사각형의 두 꼭지점(좌측 상단, 우측 하단) 좌표

  `color`: 직선의 색상.  (B,G,R) 튜플을 할당하거나 정수값을 넣는다.

  `thickness`: line의 두께.  default는 1 pixel

  `lineType` : line의 종류.  default는 cv2.LINE_8

  `shift` : 그리기 좌표 값의 축소 비율. default는 0 (일반적으로는 잘 사용 안함)



##### draw_label_info

```python
def draw_label_info(frame, x_min, y_min, label, confidence, color):
      text = label + ' ' + str('%.3f' % confidence)
        # ex) cat 0.78%
      bottomLeftCornerOfText = (x_min, y_min)
      font = cv2.FONT_HERSHEY_SIMPLEX
      fontScale = 0.8
      fontColor = color
      lineType = 2

      cv2.putText(frame, text,
                  bottomLeftCornerOfText,
                  font,
                  fontScale,
                  fontColor,
                  lineType)
```

- `cv2.putText(img, text, org, fontFace, fontScale, color, thickness = , lineType =, bottomLeftOrigin = )`

  `text` : 출력할 문자열

  `org` : text의 위치 좌측 하단 좌표

  `fontFace` : font 종류  

  `fontScale` : font size

  `bottomLeftOrigin` : Ture or False.

  `lineType` : 선형 타입.  cv2.LINE_AA 또는 2

  





#### find_max_confidence_bounding_box

Implement abbreviated version non-*maximum* suppression

```python
def find_max_confidence_bounding_box(bounding_box_info_list):
  bounding_box_info_list_sorted = sorted(bounding_box_info_list,
                                                   key=itemgetter('confidence'),
                                                   reverse=True)
# 전체 bounding box 내림차순 sorting
  max_confidence_bounding_box = bounding_box_info_list_sorted[0]

  return max_confidence_bounding_box
# confidence가 가장 높은 bounding box return
```



#### yolo_format_to_bounding_box_dict

predict bounding box info

```python
def yolo_format_to_bounding_box_dict(xcenter, ycenter, box_w, box_h, class_name, confidence):
  bounding_box_info = {}
  bounding_box_info['left'] = int(xcenter - (box_w / 2))
  bounding_box_info['top'] = int(ycenter - (box_h / 2))
  bounding_box_info['right'] = int(xcenter + (box_w / 2))
  bounding_box_info['bottom'] = int(ycenter + (box_h / 2))
  bounding_box_info['class_name'] = class_name
  bounding_box_info['confidence'] = confidence

  return bounding_box_info
```

the zore corrdinate of image located (left, top). 

so `'top' = ycenter - (box_h / 2)`  and `'bottom' = ycenter + (box_h / 2)`





#### intersection of union

- Reference : https://github.com/nilboy/tensorflow-yolo/blob/python2.7/yolo/net/yolo_tiny_net.py#L105

```python
def iou(yolo_pred_boxes, ground_truth_boxes):
      boxes1 = yolo_pred_boxes
      boxes2 = ground_truth_boxes

      boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                         boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2, boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
      boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])
      boxes2 = tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
                         boxes2[0] + boxes2[2] / 2, boxes2[1] + boxes2[3] / 2])
      boxes2 = tf.cast(boxes2, tf.float32)

      # calculate the left up point
      lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
      rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

      # intersection
      intersection = rd - lu

      inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]

      mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(intersection[:, :, :, 1] > 0, tf.float32)

      inter_square = mask * inter_square
      # overlapping parts of two square

      # calculate the boxs1 square and boxs2 square
      square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
      square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])
    
      iou = inter_square / (square1 + square2 - inter_square + 1e-6)
      # '1e-6' : for the denominator to not zero

      return iou
       
      # return value range is 0 ~ 1

```

- Args

  - yolo_pred_boxes.shape

    `[CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]`  

    `4` ====> `(x_center, y_center, w, h)`

  - ground_truth_boxes.shape

    `[4] ===> (x_center, y_center, w, h)`

  - iou

    `[CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]`



#### generate_color

Generate each colors about each class for drawing bounding boxes 

Each color is determined randomly

- Reference : https://github.com/qqwweee/keras-yolo3/blob/e6598d13c703029b2686bc2eb8d5c09badf42992/yolo.py#L82

```python
def generate_color(num_classes):
      hsv_tuples = [(x / num_classes, 1., 1.)
                    for x in range(num_classes)]
      colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
      colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
      np.random.seed(10101)  # Fixed seed for consistent colors across runs.
      np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
      np.random.seed(None)  # Reset seed to default.

      return colors
```



