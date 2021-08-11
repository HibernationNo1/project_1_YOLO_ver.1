- Bbox에 표현하는 confidence score을 predicted object confidence를 그대로 사용하고 있는 것을 확인, 계산 과정을 수정했다.

  기존의 code는 [7, 7, 2]의 predicted object confidence를 그대로 사용하지만 이는 다수의 object를 detection할 때, 기준으로 삼는 confidence score의 값은 단순히 크지만 전혀 엉뚱한 위치의 object를 가리키는 Bbox를 표현하게 된다. 

  이를 방지하기 위해 iou의 개념을 통해 실제 object가 존재하는 위치에 가까울수록 donfidence score가 높을 수 있도록 confidence score = (intersection_of_union) * (predited class probability) 으로 표현했다.

  또한 이를 위해서 각 object마다 [cell_size, cell_size, box_per_cell] shape의 iou를 계산하도록 했다.

- image에 여러 object가 존재해도 1개의 object만을 표현하는 것을 확인, 조건에 맞는 다수의 object를 detection한다면 모두 표현하도록 수정했다. 



### improvement

#### confidence score

evaluate.py 의 `main` function에서 confidence를 계산하는 code와 multi object detection을 위해 각 object마다 [cell_size, cell_size, box_per_cell] shape의 iou를 계산하는 과정을 추가했다.

train.py 의 `save_validation_result` 에 부분적으로 code의 수정을 주었다.



> 두 file의 code는 동일함

- 변경 전 `main`

  ```python
    for image_num, features in enumerate(test_data):
      batch_image = features['image']
      batch_bbox = features['objects']['bbox']
      batch_labels = features['objects']['label']
  
      batch_image = tf.squeeze(batch_image, axis=1)
      batch_bbox = tf.squeeze(batch_bbox, axis=1)
      batch_labels = tf.squeeze(batch_labels, axis=1)
  
      image, labels, object_num = process_each_ground_truth(batch_image[0], batch_bbox[0], batch_labels[0], input_width, input_height)
  
      drawing_image = image
      image = tf.expand_dims(image, axis=0)
  
      predict = YOLOv1_model(image)
      predict = reshape_yolo_preds(predict)
  
      predict_boxes = predict[0, :, :, num_classes + boxes_per_cell:]
      predict_boxes = tf.reshape(predict_boxes, [cell_size, cell_size, boxes_per_cell, 4])
  
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
  
            # add bounding box dict list
            bounding_box_info_list.append(yolo_format_to_bounding_box_dict(pred_xcenter, pred_ycenter, pred_box_w, pred_box_h, pred_class_name, pred_confidence))
  
      # make ground truth bounding box list
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
      for ground_truth_bounding_box_info in ground_truth_bounding_box_info_list:
        draw_bounding_box_and_label_info(
          ground_truth_drawing_image,
          ground_truth_bounding_box_info['left'],
          ground_truth_bounding_box_info['top'],
          ground_truth_bounding_box_info['right'],
          ground_truth_bounding_box_info['bottom'],
          ground_truth_bounding_box_info['class_name'],
          ground_truth_bounding_box_info['confidence'],
          color_list[cat_class_to_label_dict[ground_truth_bounding_box_info['class_name']]]
        )
  
      # find one max confidence bounding box
      max_confidence_bounding_box = find_max_confidence_bounding_box(bounding_box_info_list)
  
      # draw prediction
      draw_bounding_box_and_label_info(
        drawing_image,
        max_confidence_bounding_box['left'],
        max_confidence_bounding_box['top'],
        max_confidence_bounding_box['right'],
        max_confidence_bounding_box['bottom'],
        max_confidence_bounding_box['class_name'],
        max_confidence_bounding_box['confidence'],
        color_list[cat_class_to_label_dict[max_confidence_bounding_box['class_name']]]
      )
  
      # left : ground-truth, right : prediction
      drawing_image = np.concatenate((ground_truth_drawing_image, drawing_image), axis=1)
  
      # save test prediction result to png file
      if not os.path.exists(os.path.join(os.getcwd(), FLAGS.test_dir)):
        os.mkdir(os.path.join(os.getcwd(), FLAGS.test_dir))
      output_image_name = os.path.join(os.getcwd(), FLAGS.test_dir, str(int(image_num)) +'_result.png')
      cv2.imwrite(output_image_name, cv2.cvtColor(drawing_image, cv2.COLOR_BGR2RGB))
      print(output_image_name + ' saved!')
  ```

  

- 변경 후 `calculate_confidence_score`

  ```python
  	for image_num, features in enumerate(test_data):
  		batch_image = features['image']
  		batch_bbox = features['objects']['bbox']
  		batch_labels = features['objects']['label']
  
  		batch_image = tf.squeeze(batch_image, axis=1)
  		batch_bbox = tf.squeeze(batch_bbox, axis=1)
  		batch_labels = tf.squeeze(batch_labels, axis=1)
  
  		batch_bbox, batch_labels = remove_irrelevant_label(batch_bbox, batch_labels, class_name_dict)
  
  		image, labels, object_num = process_each_ground_truth(batch_image[0],
  															  batch_bbox[0], 
  															  batch_labels[0], 
  															  input_width, 
  															  input_height)
  
  		ground_truth_bounding_box_info_list = list() 	# make ground truth bounding box list
  		for each_object_num in range(object_num):
  			# make label image info list 
  			labels = np.array(labels)
  			label = labels[each_object_num, :]
  			xcenter = label[0]
  			ycenter = label[1]
  			box_w = label[2]
  			box_h = label[3]
  
  			# [1., 0.] 일 때 index_one == 0, [0., 1.] 일 때 index_one == 1
  			class_label_index = tf.argmax(label[4])
  			
  			# add ground-turth bounding box dict list
  			# 특정 class에만 ground truth bounding box information을 draw
  			for label_num in range(num_classes):
  				if int(class_label_index) == label_num:     
  					ground_truth_bounding_box_info_list.append(
  						yolo_format_to_bounding_box_dict(xcenter, ycenter, box_w, box_h,
  						 str(label_to_class_dict[label_num]), 1.0, 1.0))
  
  		drawing_image = image
  		image = tf.expand_dims(image, axis=0)
  
  		predict = YOLOv1_model(image)
  
  		predict_boxes = predict[2]
  		predict_boxes = tf.squeeze(predict_boxes, [0])
  
  		pred_P = tf.nn.softmax(predict[0])
  		pred_P = tf.squeeze(pred_P, [0])
  
  		class_prediction = pred_P  
  		class_prediction = tf.argmax(class_prediction, axis=2)
  
  		iou_predict_truth_list = list() 	# object_num, 7, 7, 2, 
  		index_class_name_list = list() 		# object_num
  		for each_object_num in range(object_num):
  			iou_predict_truth_list.append(iou(predict_boxes, labels[each_object_num, 0:4]))
  			index_class_name_list.append(tf.argmax(labels[each_object_num, 4]))
  															  
  		
  		bounding_box_info_list = list()					# make prediction bounding box list
  		for i in range(cell_size):
  			for j in range(cell_size):
  				for k in range(boxes_per_cell):
  					pred_xcenter = predict_boxes[i][j][k][0]
  					pred_ycenter = predict_boxes[i][j][k][1]
  					pred_box_w = tf.minimum(input_width * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][2]))
  					pred_box_h = tf.minimum(input_height * 1.0, tf.maximum(0.0, predict_boxes[i][j][k][3]))
  
  					for each_object_num in range(object_num):
  						pred_class_name = label_to_class_dict[index_class_name_list[each_object_num].numpy()]
  						confidence_score = pred_P[i][j][index_class_name_list[each_object_num]]
  						computed_iou = iou_predict_truth_list[each_object_num][i][j][k]
  				
  						bounding_box_info_list.append(yolo_format_to_bounding_box_dict(pred_xcenter, 
  																					   pred_ycenter,
  																					   pred_box_w,
  																					   pred_box_h,
  																					   pred_class_name,
  																					   confidence_score,
  																					   computed_iou
  																					   ))
  
  
  		ground_truth_drawing_image = drawing_image.copy()
  		# draw ground-truth image
  		for ground_truth_bounding_box_info in ground_truth_bounding_box_info_list:
  			draw_bounding_box_and_label_info(
  			ground_truth_drawing_image,
  			ground_truth_bounding_box_info['left'],
  			ground_truth_bounding_box_info['top'],
  			ground_truth_bounding_box_info['right'],
  			ground_truth_bounding_box_info['bottom'],
  			ground_truth_bounding_box_info['class_name'],
  			ground_truth_bounding_box_info['confidence_score'],
  			color_list[cat_class_to_label_dict[ground_truth_bounding_box_info['class_name']]]
  			)
  
  		# find one max confidence bounding box
  		confidence_bounding_box_list, check = find_confidence_bounding_box(bounding_box_info_list, confidence_threshold)
  
  		# draw prediction (image 위에 bounding box 표현)
  		for confidence_bounding_box in confidence_bounding_box_list:
  			draw_bounding_box_and_label_info(
  				drawing_image,
  				confidence_bounding_box['left'],
  				confidence_bounding_box['top'],
  				confidence_bounding_box['right'],
  				confidence_bounding_box['bottom'],
  				confidence_bounding_box['class_name'],
  				confidence_bounding_box['confidence_score'],
  				color_list[cat_class_to_label_dict[confidence_bounding_box['class_name']]]) 
  
  
  		# left : ground-truth, right : prediction
  		drawing_image = np.concatenate((ground_truth_drawing_image, drawing_image), axis=1)
  		print(f"image index: {image_num}")
  
  		# save test prediction result to png file
  		if not os.path.exists(os.path.join(os.getcwd(), dir_name, FLAGS.test_dir)):
  			os.mkdir(os.path.join(os.getcwd(), dir_name, FLAGS.test_dir))
  		output_image_name = os.path.join(os.getcwd(), dir_name, FLAGS.test_dir, str(int(image_num)) +'_result.png')
  		if check:
  			print("Detection success")
  			cv2.imwrite(output_image_name, cv2.cvtColor(drawing_image, cv2.COLOR_BGR2RGB))
  			print(output_image_name + ' saved! \n')
  		else : 
  			print("Detection failed \n")
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

