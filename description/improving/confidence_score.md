Bbox에 표현하는 confidence score을 predicted object confidence를 그대로 사용하고 있는 것을 확인. 





### Improving



confidence score = (predicted object confidence) * (class probability)/10 로 표현했다.

> 10을 나눠주는 이유 : class_loss의 정확한 학습을 위해 class probability label값을 0~1 의 scale에서 0~10로 변경했기 때문에 class probability를 다시 scale을 0~1로 낮춰주기 위함이다.



train.py 의 `save_validation_result` function과 

evaluate.py 의 `main` function에 부분적으로 code의 수정을 주었다.



> 두 file의 code는 동일함

- 변경 전

  ```python
  		confidence_boxes = predict[0, :, :, num_classes:num_classes + boxes_per_cell]
  		confidence_boxes = tf.reshape(confidence_boxes, [cell_size, cell_size, boxes_per_cell, 1])
  
  		class_prediction = predict[0, :, :, 0:num_classes]
  		class_prediction = tf.argmax(class_prediction, axis=2)
  
  		bounding_box_info_list = []
  		for i in range(cell_size):
  			for j in range(cell_size):
  				for k in range(boxes_per_cell):
  					pred_xcenter = predict_boxes[i][j][k][0]
  					pred_ycenter = predict_boxes[i][j][k][1]
  					pred_box_w = tf.minimum(input_width * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][2]))
  					pred_box_h = tf.minimum(input_height * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][3]))
  
  					pred_class_name = cat_label_to_class_dict[class_prediction[i][j].numpy()]
  					pred_confidence = confidence_boxes[i][j][k].numpy()
  ```

  

- 변경 후

  ```python
  		confidence_boxes = predict[0, :, :, num_classes:num_classes + boxes_per_cell]
  		confidence_boxes = tf.reshape(confidence_boxes, [cell_size, cell_size, boxes_per_cell, 1])
  
  		# 각 셀마다 class probability가 가장 높은 prediction value의 index추출(predict한 class name)
  		# 0:num_class는 에는 각 class에 대한 predicted probability value가 있다.(class 확률의 합 = 1)
  		class_prediction = predict[0, :, :, 0:num_classes]  
  		class_prediction_value = tf.reduce_max(class_prediction, axis = 2) # for compute confidence_score
  		class_prediction = tf.argmax(class_prediction, axis=2)
  
  		confidence_score = np.zeros_like(confidence_boxes[:, :, :, 0])
  		for i in range(boxes_per_cell):
  			confidence_score[:, :, i] = (confidence_boxes[:, :, i, 0] * class_prediction_value)/10
  		
  		# make prediction bounding box list
  		bounding_box_info_list = []
  		for i in range(cell_size):
  			for j in range(cell_size):
  				for k in range(boxes_per_cell):
  					pred_xcenter = predict_boxes[i][j][k][0]
  					pred_ycenter = predict_boxes[i][j][k][1]
  					pred_box_w = tf.minimum(input_width * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][2]))
  					pred_box_h = tf.minimum(input_height * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][3]))
  				   
  					
  					pred_class_name = cat_label_dict[class_prediction[i][j].numpy()]                   
  					pred_confidence = confidence_score[i][j][k]
  ```

  

