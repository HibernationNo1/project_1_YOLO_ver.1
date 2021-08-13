**single object detection에서 multi object detection으로 기능 확장**



### todo list

- [extract multi class](#'extract multi class ')

  dataset에서 1개의 class만 extraction하는 것에서 여러개의 class를 extraction하는 것으로 변경

- [detect multi object](#'detect multi object')

  label과 predicted image통해 진행되는 test 결과를 확인하는 과정에서 1개의 object만을 표현하는 것을 다수의 object를 표현하도록 변경 





### improvement



#### extract multi class 

dataset.py의 load_pascal_voc_dataset에서 사용된 filter함수를 위해 정의한 predicate를 통해 기능 변경

1. dataset에서 가져오는 class를 cat에서 cow로 변경하기

```python
def predicate(x):  # x는 전체 dataset
	label = x['objects']['label']

	# 7또는 9라는 label의 object가 하나라도 포함 된 data는 모두 추려낸다.	
	isallowed_cow = tf.equal(tf.constant([9.0]), tf.cast(label, tf.float32)) 	# label이 9인 element만 True

	reduced = tf.reduce_sum(tf.cast(isallowed_cow, tf.float32))

	return tf.greater(reduced, tf.constant(0.))  # label이 7인 element의 개수가 0보다 클 때(1개 이상일때) True

```



2. dataset에서 cat, cow 두 개의 class를 가져오기 

   ```python
   def predicate(x):  # x는 전체 dataset
   	label = x['objects']['label']
   
   	# 7또는 9라는 label의 object가 하나라도 포함 된 data는 모두 추려낸다.	
   	isallowed_cat = tf.equal(tf.constant([7.0]), tf.cast(label, tf.float32)) 	# label이 7인 element만 True
   	isallowed_cow = tf.equal(tf.constant([9.0]), tf.cast(label, tf.float32)) 	# label이 9인 element만 True
   
   	reduced_cat = tf.reduce_sum(tf.cast(isallowed_cat, tf.float32)) 			# label이 7인 element의 개수
   	reduced_cow = tf.reduce_sum(tf.cast(isallowed_cow, tf.float32))
   	reduced = reduced_cat + reduced_cow  # cat과 cow data 합산
   
   	return tf.greater(reduced, tf.constant(0.))  # label이 7인 element의 개수가 0보다 클 때(1개 이상일때) True
   
   ```

   



**최종**

3. dataset에서 label dictionary를 만들어 key값을 통해 특정 label이 포함된 모든 data를 추려내도록

   dataset.py의 전역에 감지할 클래스 목록을 dictionary로 생성

   ```python
   # dict of classes to detect 
   class_name_dict = {
   	7: "cat", 9:"cow"
   }
   ```

   

   define predicate

   ```python
   def predicate(x):  # x는 전체 dataset
   	label = x['objects']['label']
   	
   	# 7또는 9라는 label의 object가 하나라도 포함 된 data는 모두 추려낸다.	
   	reduced_sum = 0.0
   
   	for label_num in class_name_dict.keys():
   		isallowed = tf.equal(tf.constant([float(label_num)]), tf.cast(label, tf.float32)) # label이 label_num인 element만 True
   		reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32)) 	# label이 class_num인 element의 개수
   		reduced_sum += reduced
   
   	return tf.greater(reduced_sum, tf.constant(0.))  # label이 7인 element의 개수가 0보다 클 때(1개 이상일때) True
   
   ```

   

#### detect multi object

evaluate.py에서

ground truth image에 label Bbox와 prediction prediction Bbox를 concaterate하여 비교하는 image를 만들어내는 과정에서

각 object에 대해서 iou를 계산하고, 이러한 정보를 저장하는 list에 할당.

**변경 전**

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



**변경 후**

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

		if image_num == 200:
			import sys
			sys.exit()

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

