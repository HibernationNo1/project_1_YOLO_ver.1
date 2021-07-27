Bbox에 표현하는 confidence score을 predicted object confidence를 그대로 사용하고 있는 것을 확인. 



### Improving

confidence score = (predicted object confidence) * (best predited class probability) 로 표현했다.



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
  		pred_C = predict[1]
  		pred_C = tf.squeeze(pred_C)
  		pred_P = predict[0]
  		pred_P = tf.squeeze(pred_P)
  		# pred_C : 예측한 Bbox영역 안에 object가 있을 probability
  		# pred_P : 각 class에 대한 predicted probability
  		# 각 셀마다 class probability가 가장 높은 prediction value의 index추출(predict한 class name)
  		class_prediction = pred_P  
  		class_prediction = tf.argmax(class_prediction, axis=2)
  
  		# 각 cell에 위치한 각 Bbox의 confidence score 계산
  		# confidence_score = predicted object confidence * best predited class probability
  		confidence_score = np.zeros_like(pred_C)
  		max__probability = tf.expand_dims(tf.reduce_max(pred_P, axis = 2), axis=2)
   
  		confidence_score = pred_C * max__probability
  		
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
  
  

가장 큰 confidence_score를 기록해보았다.

utils.py의 `find_enough_confidence_bounding_box` 수정

```python
def find_enough_confidence_bounding_box(bounding_box_info_list, validation_image_index):
	bounding_box_info_list_sorted = sorted(bounding_box_info_list,
											key=itemgetter('confidence_score'),
											reverse=True)
	confidence_bounding_box_list = list()

	# 가장 큰 confidence_score를 출력
	print(f'image index:{validation_image_index},  best_confidence_score: {bounding_box_info_list_sorted[0]["confidence_score"]}')
	
	# confidence값이 0.5 이상인 Bbox는 모두 표현
	for index, features in enumerate(bounding_box_info_list_sorted):
		if bounding_box_info_list_sorted[index]['confidence_score'] > 0.5:
			confidence_bounding_box_list.append(bounding_box_info_list_sorted[index])

	return confidence_bounding_box_list
```

