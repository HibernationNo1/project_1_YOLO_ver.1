validation, test image 비교에서 Bbox를 표시할 때 1개만 하는것이 아닌, confidence가 충분히 높은 object는 모두 Bbox로 표시하도록 했다.



기존의 `find_max_confidence_bounding_box`라는 function에서 `find_enough_confidence_bounding_box` 라는 function으로 code를 수정했다.



**변경 전 find_max_confidence_bounding_box**

```python
def find_max_confidence_bounding_box(bounding_box_info_list):
  bounding_box_info_list_sorted = sorted(bounding_box_info_list,
                                                   key=itemgetter('confidence'),
                                                   reverse=True)
  max_confidence_bounding_box = bounding_box_info_list_sorted[0]

  return max_confidence_bounding_box
```

> 가장 높은 confidence를 가진 Bbox 1개만 추려낸다.



 **변경 후 find_enough_confidence_bounding_box**

```python
def find_enough_confidence_bounding_box(bounding_box_info_list):
	bounding_box_info_list_sorted = sorted(bounding_box_info_list,
											key=itemgetter('confidence'),
											reverse=True)
	confidence_bounding_box_list = list()

	# confidence값이 0.5 이상인 Bbox는 모두 표현
	for index, features in enumerate(bounding_box_info_list_sorted):
		if bounding_box_info_list_sorted[index]['confidence'] > 0.5:
			confidence_bounding_box_list.append(bounding_box_info_list_sorted[index])

	return confidence_bounding_box_list
```

> confidence값이 0.5 이상인 모든 Bbox를 추려낸다.



main.py 의 `save_validation_result` function과 evaluate.py의 `main` function에서 

for문을 통해 Bbox를 draw해야 한다.

main.py 의 `save_validation_result`

```python
for confidence_bounding_box in confidence_bounding_box_list:
	draw_bounding_box_and_label_info(
				drawing_image,
				confidence_bounding_box['left'],
				confidence_bounding_box['top'],
				confidence_bounding_box['right'],
				confidence_bounding_box['bottom'],
				confidence_bounding_box['class_name'],
				confidence_bounding_box['confidence'],
				color_list[cat_class_to_label_dict[confidence_bounding_box['class_name']]])
```

