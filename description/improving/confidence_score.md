- Bbox에 표현하는 confidence score을 predicted object confidence를 그대로 사용하고 있는 것을 확인, 계산 과정을 수정했다.

  기존의 code는 [7, 7, 2]의 predicted object confidence를 그대로 사용하지만 이는 다수의 object를 detection할 때, 기준으로 삼는 confidence score의 값은 단순히 크지만 전혀 엉뚱한 위치의 object를 가리키는 Bbox를 표현하게 된다. 

  이를 방지하기 위해 iou의 개념을 통해 실제 object가 존재하는 위치에 가까울수록 donfidence score가 높을 수 있도록 confidence score = (intersection_of_union) * (predited class probability) 으로 표현했다.

  또한 이를 위해서 각 object마다 [cell_size, cell_size, box_per_cell] shape의 iou를 계산하도록 했다.

- image에 여러 object가 존재해도 1개의 object만을 표현하는 것을 확인, 조건에 맞는 다수의 object를 detection한다면 모두 표현하도록 수정했다. 



### Improving

#### confidence score

train.py 의 `save_validation_result` 에서 confidence를 계산하는 code를 추가했다.

evaluate.py 의 `main` function에 부분적으로 code의 수정을 주었다.



> 두 file의 code는 동일함

- 변경 전 `calculate_confidence_score`

  ```python
  		predict = model(image)
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

  

- 변경 후 `calculate_confidence_score`

  ```python
  		predict = model(image)
  		# predict[0] == pred_class 		[1, cell_size, cell_size, num_class]
  		# predict[1] == pred_confidence [1, cell_size, cell_size, boxes_per_cell]
  		# predict[2] == pred_coordinate	[1, cell_size, cell_size, boxes_per_cell, 4]
  		# tf.shape(predict)[0] == batch_size
  			
  		# parse prediction(x, y, w, h)
  		predict_boxes = predict[2]
  		predict_boxes = tf.squeeze(predict_boxes, [0])
  
  
  		# pred_P : 각 class에 대한 predicted probability
  		pred_P = tf.nn.softmax(predict[0])
  		pred_P = tf.squeeze(pred_P, [0])
  
  		# 각 cell마다 가장 높은 predicted class probability value의 index추출(predict한 class name)
  		class_prediction = pred_P  
  		class_prediction = tf.argmax(class_prediction, axis=2)
  
  
  		bounding_box_info_list = list()					# make prediction bounding box list
  		ground_truth_bounding_box_info_list = list() 	# make ground truth bounding box list
  		for each_object_num in range(object_num):
  			labels = np.array(labels)
  			label = labels[each_object_num, :]
  			xcenter = label[0]
  			ycenter = label[1]
  			box_w = label[2]
  			box_h = label[3]
  			class_label = label[4]
  			
  			# [1., 0.] 일 때 index_one == 0, [0., 1.] 일 때 index_one == 1
  			index_one = tf.argmax(class_label, axis = 0)
  		  	
  			# add ground-turth bounding box dict list
  			# 특정 class에만 ground truth bounding box information을 draw
  			for label_num in range(num_classes):
  				if int(index_one) == label_num:     
  					ground_truth_bounding_box_info_list.append(
  						yolo_format_to_bounding_box_dict(xcenter, ycenter, box_w, box_h,
  						 str(label_to_class_dict[label_num]), 1.0))
  
  			# add prediction bounding box dict list
  			for i in range(cell_size):
  				for j in range(cell_size):
  					for k in range(boxes_per_cell):
  						pred_xcenter = predict_boxes[i][j][k][0]
  						pred_ycenter = predict_boxes[i][j][k][1]
  						pred_box_w = tf.minimum(input_width * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][2]))
  						pred_box_h = tf.minimum(input_height * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][3]))
  					
  						pred_class_name = label_to_class_dict[class_prediction[i][j].numpy()]   
  
  						iou_predict_truth = iou(predict_boxes, label[0:4])
                          
                   
  						# confedence_score = class_probability * intersection_of_union
                     	 	# class_probability는 여러 class중 실제 정답에 대해 예측한 확률값을 사용한다.
  						confidence_score = pred_P[i][j][class_label_index] * iou_predict_truth[i][j][k]
                      
  						# for문이 끝나면 bounding_box_info_list에는 (object_num * cell_size * cell_size * box_per_cell)개의 bounding box의 information이 들어있다.
  						# 각 bounding box의 information은 (x, y, w, h, class_name, confidence_score)이다.
  						# add bounding box dict list
  						bounding_box_info_list.append(yolo_format_to_bounding_box_dict(pred_xcenter, 
  																				   pred_ycenter,
  																				   pred_box_w,
  																				   pred_box_h,
  																				   pred_class_name,
  																				   confidence_score))
  ```
  





#### display multi Bbox

utils.py의 `find_enough_confidence_bounding_box` 수정



**변경 전 `find_max_confidence_bounding_box`**

```python
def find_max_confidence_bounding_box(bounding_box_info_list):
  bounding_box_info_list_sorted = sorted(bounding_box_info_list,
                                                   key=itemgetter('confidence'),
                                                   reverse=True)
  max_confidence_bounding_box = bounding_box_info_list_sorted[0]

  return max_confidence_bounding_box
```

> 가장 높은 confidence를 가진 Bbox 1개만 추려낸다.



 **변경 후 `find_confidence_bounding_box`**

```python
def find_confidence_bounding_box(bounding_box_info_list, confidence_threshold):
	bounding_box_info_list_sorted = sorted(bounding_box_info_list,
											key=itemgetter('confidence_score'),
											reverse=True)
	confidence_bounding_box_list = list()

	# confidence값이 confidence_threshold 이상인 Bbox는 모두 표현
	for index in range(len(bounding_box_info_list_sorted)):
		if bounding_box_info_list_sorted[index]['confidence_score'] > confidence_threshold:
			confidence_bounding_box_list.append(bounding_box_info_list_sorted[index])
			print(f"confidence_score : {bounding_box_info_list_sorted[index]['confidence_score']:.2f}")
		else : 
			break

	return confidence_bounding_box_list
```

>`confidence_threshold` 값은 train.py에서 정의되었다.
>
>

여긴 나중에 `num_display_Bbox`를 `len(bounding_box_info_list_sorted)` 로 바꾼 후

confidence_threshold > 0.8 이상인것만 확인되도록 함 해보자.

