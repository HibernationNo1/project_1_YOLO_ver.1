**performance evaluation 기능을 추가했다.**



validation결과에 대한 성능 평가 기능을 추가했다.

성능 평가는 세 가지 경우를 고려했다.

- [average_detection_rate](#average_detection_rate)

- [perfect_detection_accuracy](#perfect_detection_accuracy)
- [classification_accuracy](#classification_accuracy)



[구현 코드](function)



### improving



#### average_detection_rate

vaildation data에 대한 평균 object detection 비율이다.

Performance Evaluation Index 중 **Recall**의 방법을 따랐다.


$$
Detection\ Rate = \frac{Num\ Detected\ Object}{Num\ Label\ Object} * 100%
$$

$$
Average\ Detection\ Rate = \frac{Sum \ Detection\ Rate }{Num\ Visualize\ Image}
$$


average_detection_rate = (sum of detection rate / number of validation image)



#### perfect_detection_accuracy

vaildation data에 대한 완벽한 object detection이 이루어진 비율이다. 


$$
Perfect\ Detection\ Accuracy = \frac{Num\ Perfect\ Detection }{Num\ Visualize\ Image}
$$
perfect_detection : detection_rate == 100% 인 경우

perfect_detection_accuracy =  (number of perfect detection / number of validation image)



#### classification_accuracy

vaildation data에 대한 정확한 classification이 이루어진 비율이다.

perfect detection이라는 전제 조건에서 성공적인 classification가 이루어졌는지 확인했다.

즉, perfect detection인 경우가 아니면 success classification 확인 과정을 수행하지 않았다. 


$$
Classification Accuracy = \frac{Num\ Correct\ Answers\ Class }{Num\ Visualize\ Image}
$$


classification_accuracy = (number of case that correct answers about class / number of validation image)



**success classification 확인 과정**

perfect detection의 조건이면 detected object의 개수가 label과 predict가 동일할 것이다.

1. label과 prediction의 object list를 x좌표 기준으로 올림차순 정렬을 수행한다.

2. x좌표가 낮은 object부터 x좌표가 높은 object 순으로 label과 prediction의 class name이 동일한지 확인한다.
3. 2번의 조건이 만족하면, label과 prediction의 object list를 y좌표 기준으로 올림차순 정렬을 수행한다.
4. y좌표가 낮은 object부터 y좌표가 높은 object 순으로 label과 prediction의 class name이 동일한지 확인한다.
5. 1, 2, 3, 4번의 동작에서 모든 조건에 부합한 경우라면, success classification인 것으로 간주한다.



#### function

utils.py의 위치에 define했으며 함수의 반환을 분모 삼아 성능평가 계산은 main.py의 `save_validation_result` function에서 진행한다.



##### performance_evaluation

perfect rate를 계산하고 perfect detection여부를 확인한다.

perfect detection인 경우에는 success classification여부를 확인한다.

```python
def performance_evaluation(confidence_bounding_box_list, object_num, labels, class_name_to_label_dict):

	x_center_sort_labels = None
	y_center_sort_labels = None
	x_center_sort_pred = None
	y_center_sort_pred = None

	pred_list = np.zeros(shape =(object_num, 3))

	correct_answers_class_num = 0.0  # classification accuracy 계산을 위한 값
	success_detection_num = 0.0 # perfect detection accuracy 계산을 위한 값


	# label object 중 detection한 object의 비율
	detection_rate = len(confidence_bounding_box_list)/object_num

	if detection_rate == 1: # label과 같은 수의 object를 detection했을 때
		success_detection_num +=1
		print(f"detection_rate = {detection_rate}")

		# detection_rate == 100% 일 때 correct_answers_class_num 계산 
		for each_object_num in range(object_num): 
			label = labels[each_object_num, :] 
			
			confidence_bounding_box = confidence_bounding_box_list[each_object_num]
			# compute x, y center coordinate 
			xcenter = int((confidence_bounding_box['left'] + confidence_bounding_box['right'] - 1.0) /2) # 1.0은 int()감안
			ycenter = int((confidence_bounding_box['top'] + confidence_bounding_box['bottom'] - 1.0) /2) 
			
			pred_list[each_object_num][0] = xcenter
			pred_list[each_object_num][1] = ycenter
			pred_list[each_object_num][2] = class_name_to_label_dict[str(confidence_bounding_box_list[0]['class_name'])] # pred_class_num

		if object_num == 1:
			# label class와 예측한 class가 같다면
			if int(label[0][4]) == class_name_to_label_dict[str(confidence_bounding_box_list[0]['class_name'])]:
				correct_answers_class_num +=1
		else:  # image에 object가 2개 이상일 때
			x_center_sort_labels = x_y_center_sort(labels, "x") # x좌표 기준으로 정렬한 labels
			y_center_sort_labels = x_y_center_sort(labels, "y") # y좌표 기준으로 정렬한 labels
			x_center_sort_pred_list = x_y_center_sort(pred_list, "x")  	# x좌표 기준으로 정렬한 pred_list
			y_center_sort_pred_list = x_y_center_sort(pred_list, "y")	# y좌표 기준으로 정렬한 pred_list

			# x좌표가 낮은 위치의 image부터 큰 위치의 image까지 detected image의 class가 동일지 확인
			for x_each_object_num in range(object_num): 
				x_center_sort_label = x_center_sort_labels[x_each_object_num, :]
				x_center_sort_pred = x_center_sort_pred_list[x_each_object_num, :]
				if int(x_center_sort_label[4]) == int(x_center_sort_pred[3]): # class가 동일하면 pass
					pass
				else : 
					break # 하나라도 다르면 break

				if x_each_object_num == object_num-1: # x좌표 기준으로 위 조건이 만족한다면
					# y좌표가 낮은 위치의 image부터 큰 위치의 image까지 detected image의 calss가 동일지 확인
					for y_each_object_num in range(object_num):
						y_center_sort_label = y_center_sort_labels[y_each_object_num, :]
						y_center_sort_pred = y_center_sort_pred_list[y_each_object_num, :]
						if int(y_center_sort_label[4]) == int(y_center_sort_pred[3]):
							pass
						else : 
							break # 하나라도 다르면 break	

						if x_each_object_num == object_num-1:   # y좌표 기준으로도 위 조건이 만족한다면 
							correct_answers_class_num +=1
	elif detection_rate > 1: # label보다 더 많은 object를 detection했을 때
		print("Over detection")
	else :
		print(f"detection_rate = {detection_rate}")

		
	return success_detection_num, correct_answers_class_num, detection_rate
```





##### x_y_center_sort

label과 prediction의 object list를 x(또는 y)좌표 기준으로 올림차순 정렬을 수행한다.

이 때 정렬은 x(또는 y)좌표에 대한 정렬만 수행하는 것이 아니라, original list x좌표의 index를 추적해서 class name이 저장된 element까지 같이 정렬순서에 맞춰 정렬하도록 한다. 

```python
def x_y_center_sort(labels, taget):

	tmp = np.zeros_like(labels)
	if taget == "x":
		label = list(np.array(labels[:, 0]))
	elif taget == "y":
		label = list(np.array(labels[:, 1]))

	origin_label = label.copy()
	label.sort()
	for i_index, i_value in enumerate(label):
		for j_index, j_value in enumerate(origin_label):
			if i_value == j_value:
				tmp[i_index] = labels[j_index]
				continue
	labels = tf.constant(tmp)
	
	return labels
```



**in main.py**

compute performance evaluation, write performance evaluation log on tensorboard

```python
def save_validation_result(model,
						   ckpt, 
						   validation_summary_writer,
						   average_detection_rate_writer,
						   perfect_detection_accuracy_writer,
						   classification_accuracy_writer,
						   num_visualize_image):
    
    for validation_image_index in range(num_visualize_image): # 의 계층에서
    detection_num, class_num, detection_rate = performance_evaluation(confidence_bounding_box_list, object_num, labels, class_name_to_label_dict)
	success_detection_num += detection_num
	correct_answers_class_num += class_num
	detection_rate_sum +=detection_rate
    
    
	average_detection_rate = detection_rate / num_visualize_image  				# 평균 object detection 비율	
	perfect_detection_accuracy = success_detection_num / num_visualize_image	# 완벽한 object detection이 이루어진 비율
	classification_accuracy = correct_answers_class_num / num_visualize_image 	# 정확한 classicifiation이 이루어진 비율
	
	with average_detection_rate_writer.as_default():
		tf.summary.scalar('average_detection_rate', average_detection_rate, step=int(ckpt.step))

	with perfect_detection_accuracy_writer.as_default():
		tf.summary.scalar('perfect_detection_accuracy', perfect_detection_accuracy, step=int(ckpt.step))

	with classification_accuracy_writer.as_default():
		tf.summary.scalar('classicifiation_accuracy', classification_accuracy, step=int(ckpt.step))	
    
```

