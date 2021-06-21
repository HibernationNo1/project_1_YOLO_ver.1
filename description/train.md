# train



```python
import tensorflow as tf
import numpy as np
import os
import random

from loss import yolo_loss
from dataset import process_each_ground_truth
from utils import draw_bounding_box_and_label_info, find_max_confidence_bounding_box, yolo_format_to_bounding_box_dict


def reshape_yolo_preds(preds):
    # flatten vector -> cell_size x cell_size x (num_classes + 5 * boxes_per_cell)
    return tf.reshape(preds, [tf.shape(preds)[0], cell_size, cell_size, num_classes + 5 * boxes_per_cell])

# define loss function
'''
model : instance of model class
batch_image : train data 
batch_bbox : batch bounding box
batch_labels : label data
'''
def calculate_loss(model, batch_image, batch_bbox, batch_labels):
    total_loss = 0.0
    coord_loss = 0.0
    object_loss = 0.0
    noobject_loss = 0.0
    class_loss = 0.0
    for batch_index in range(batch_image.shape[0]): # 전체 batch에 대해서 1개씩 반복
        image, labels, object_num = process_each_ground_truth(batch_image[batch_index],
                                                          batch_bbox[batch_index],
                                                          batch_labels[batch_index],
                                                          input_width, input_height)
        # process_each_ground_truth : 원하는 data를 parsing
        # image : resize된 data
        # labels : 절대 좌표
        # object_num : object 개수
    
        image = tf.expand_dims(image, axis=0)
	    # expand_dims을 사용해서 0차원에 dummy dimension 추가

        predict = model(image) # predict의 shape은 flatten vector 형태
        predict = reshape_yolo_preds(predict)
        # shape = [1, cell_size, cell_size, num_classes, 5 * boxes_per_cell]

        for object_num_index in range(object_num): # 실제 object개수만큼 for루프
            each_object_total_loss, each_object_coord_loss, each_object_object_loss, each_object_noobject_loss, each_object_class_loss = yolo_loss(predict[0],
                                   labels,
                                   object_num_index,
                                   num_classes,
                                   boxes_per_cell,
                                   cell_size,
                                   input_width,
                                   input_height,
                                   coord_scale,
                                   object_scale,
                                   noobject_scale,
                                   class_scale
                                   )
        # 각 return값은 1개의 image에 대한 여러 loss 값임

      
      
            total_loss = total_loss + each_object_total_loss
            coord_loss = coord_loss + each_object_coord_loss
            object_loss = object_loss + each_object_object_loss
            # 각각 전체의 batch에 대해서 loss 합산

        noobject_loss = noobject_loss + each_object_noobject_loss
# gradient descent을 수행하는 method      class_loss = class_loss + each_object_class_loss

    return total_loss, coord_loss, object_loss, noobject_loss, class_loss
    
                                                                                    
                                                                                    
                                                                                   
  
def train_step(optimizer, model, batch_image, batch_bbox, batch_labels): 
    with tf.GradientTape() as tape:
        total_loss, coord_loss, object_loss, noobject_loss, class_loss = calculate_loss(model, batch_image, batch_bbox, batch_labels)
	# tensor board를 남기기 위해 return
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss, coord_loss, object_loss, noobject_loss, class_loss

   
def save_validation_result(model, ckpt, validation_summary_writer, num_visualize_image):
    total_validation_total_loss = 0.0
    total_validation_coord_loss = 0.0  # 전체 data의 validation data
    total_validation_object_loss = 0.0
    total_validation_noobject_loss = 0.0  # 전체 data의 
    total_validation_class_loss = 0.0
    for iter, features in enumerate(validation_data):
        batch_validation_image = features['image']
        batch_validation_bbox = features['objects']['bbox']
        batch_validation_labels = features['objects']['label']

        # validation data와 model이 predict한 값 간의 loss값 compute
        batch_validation_image = tf.squeeze(batch_validation_image, axis=1)                                                                                                                                      
        batch_validation_bbox = tf.squeeze(batch_validation_bbox, axis=1)
        batch_validation_labels = tf.squeeze(batch_validation_labels, axis=1)
    
        validation_total_loss, validation_coord_loss, validation_object_loss, validation_noobject_loss, validation_class_loss = calculate_loss(model,
																																batch_validation_image,
																																batch_validation_bbox,
																																batch_validation_labels)
       
        total_validation_total_loss = total_validation_total_loss + validation_total_loss
        total_validation_coord_loss = total_validation_coord_loss + validation_coord_loss
        total_validation_object_loss = total_validation_object_loss + validation_object_loss
        total_validation_noobject_loss = total_validation_noobject_loss + validation_noobject_loss
        total_validation_class_loss = total_validation_class_loss + validation_class_loss
      
  #     save validation tensorboard log
    with validation_summary_writer.as_default():
        tf.summary.scalar('total_validation_total_loss', total_validation_total_loss, step=int(ckpt.step))
        tf.summary.scalar('total_validation_coord_loss', total_validation_coord_loss, step=int(ckpt.step))
        tf.summary.scalar('total_validation_object_loss ', total_validation_object_loss, step=int(ckpt.step))
        tf.summary.scalar('total_validation_noobject_loss ', total_validation_noobject_loss, step=int(ckpt.step))
        tf.summary.scalar('total_validation_class_loss ', total_validation_class_loss, step=int(ckpt.step))
      
  # save validation test image
    for validation_image_index in range(num_visualize_image):
        random_idx = random.r # resize된 YOLO 원본 input imageandint(0, batch_validation_image.shape[0] - 1)
        image, labels, object_num = process_each_ground_truth(batch_validation_image[random_idx],
															  batch_validation_bbox[random_idx],
                                                              batch_validation_labels[random_idx],
															  input_width, input_height)
    
        drawing_image = image
  
        image = tf.expand_dims(image, axis=0)
        predict = model(image)
        predict = reshape_yolo_preds(predict)
    
        # parse prediction
        predict_boxes = predict[0, :, :, num_classes + boxes_per_cell:]
        predict_boxes = tf.reshape(predict_boxes,
						[cell_size, cell_size, boxes_per_cell, 4])
    
        confidence_boxes = predict[0, :, :, num_classes:num_classes + boxes_per_cell]
        confidence_boxes = tf.reshape(confidence_boxes,
									 [cell_size, cell_size, boxes_per_cell, 1])
       
        # Non-maximum suppression
        class_prediction = predict[0, :, :, 0:num_classes]
        class_prediction = tf.argmax(class_prediction, axis=2)
          
           # ma ke prediction bounding box list
        bounding_box_info_list = []
        for i in range(cell_size):
            for j in range(cell_size):
                for k in range(boxes_per_cell):
                    pred_xcenter = predict_boxes[i][j][k][0]
                    pred_ycenter = predict_boxes[i][j][k][1]
                    pred_box_w = tf.minimum(input_width * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][2]))
                    pred_box_h = tf.minimum(input_height * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][3]))
                   
                    
                    pred_class_name = cat_label_dict[class_prediction[i][j].numpy()]                                                       
                    pred_confidence = confidence_boxes[i][j][k].numpy()[0]
              # for문이 끝나면 bounding_box_info_list에는 7(cell_size) * 7(cell_size) * 2(boxes_per_cell) = 98 개의 bounding box의 information이 들어있다.
              # add bounding box dict list
                    bounding_box_info_list.append(yolo_format_to_bounding_box_dict(pred_xcenter, 
																				   pred_ycenter,
																				   pred_box_w,
																				   pred_box_h,
																				   pred_class_name,
																				   pred_confidence))
       
        #make ground truth bounding box list
        ground_truth_bounding_box_info_list = []
        for each_object_num in range(object_num):
            labels = np.array(labels)
            labels = labels.astype('float32')
            label = labels[each_object_num, :]
            xcenter = label[0]
            ycenter = label[1]
            box_w = label[2]
            box_h = label[3]
            class_label = label[4]
          
           # label 7 : cat
           # add ground-turth bounding box dict list
            if class_label == 7:     
                ground_truth_bounding_box_info_list.append(
                    yolo_format_to_bounding_box_dict(xcenter, ycenter, box_w, box_h, 'cat', 1.0))
        
        ground_truth_drawing_image = drawing_image.copy()
        # draw ground-truth image
	    # window에 정답값의 bounding box와 그에 따른 information을 draw
        for ground_truth_bounding_box_info in ground_truth_bounding_box_info_list:
            draw_bounding_box_and_label_info(
                ground_truth_drawing_image,
                ground_truth_bounding_box_info['left'],
                ground_truth_bounding_box_info['top'],
                ground_truth_bounding_box_info['right'],
                ground_truth_bounding_box_info['bottom'],
                ground_truth_bounding_box_info['class_name'],
                ground_truth_bounding_box_info['confidence'],
                color_list[cat_class_to_label_dict[ground_truth_bounding_box_info['class_name']]])
         
        # find one max confidence bounding box
		# Non-maximum suppression을 사용하지 않고, 약식으로 진행
        max_confidence_bounding_box = find_max_confidence_bounding_box(bounding_box_info_list)
        # confidence가 가장 큰 bounding box 하나만 선택

        # draw prediction
		# image 위에 bounding box 표현
        draw_bounding_box_and_label_info(
            dra_ing_image,
            max_confidence_bounding_box['left'],
            max_confidence_bounding_box['top'],
            max_confidence_bounding_box['right'],
            max_confidence_bounding_box['bottom'],
            max_confidence_bounding_box['class_name'],
            max_confidence_bounding_box['confidence'],
            color_list[cat_class_to_label_dict[max_confidence_bounding_box['class_name']]])
     
        

        # left : ground-truth, right : prediction
        drawing_image = np.concatenate((ground_truth_drawing_image, drawing_image), axis=1)
	    # 두 이미지를 연결(왼쪽엔 ground_truth, 오른쪽엔 drawing_image)
        drawing_image = drawing_image / 255
		drawing_image = tf.expand_dims(drawing_image, axis = 0)
	    # nomalization하고 dummy dimension 추가 아래 save tensorboard log에서 어떤 값이 write될지 확인해보자.

        # save tensorboard log
        with validation_summary_writer.as_default():
            tf.summary.image('validation_image_'+str(validation_image_index), drawing_image, step=int(ckpt.step))
```

