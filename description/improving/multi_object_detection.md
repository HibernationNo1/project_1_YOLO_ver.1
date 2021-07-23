**single object detection에서 multi object detection으로 기능 확장**



dataset.py의 load_pascal_voc_dataset에서 사용된 filter함수를 위해 정의한 predicate를 통해 기능 변경



### improvement

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

   