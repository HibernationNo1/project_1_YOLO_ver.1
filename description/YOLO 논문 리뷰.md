# YOLO 논문 리뷰

## Abstract

1. 전체 image를 한 번의 계산으로  bounding box와 class probabilities를 계산하는 neural network임을 설명

2. network의 pipeline이 end-to-end(종단간 학습)임을 설명

3. 거의 real-time에 가까운, 초당 45장의 image를 처리할 수 있음을 설명

   > Fast YOLO를 사용하면 성능은 떨어지지만 초당 155개의 frame을 계산할 수 있다.

4. state-of-the-art(가장 빠른 성능) models 와 비교할만한 성능을 가지고 있음

5. localiztion errors가 있지만 배경에 대한 false positive가 적음



## Conclusion

1. YOLO는 분류를 바탕으로 둔 접근법과는 다르게, loss를 직접적으로 계산하며 학습하기 때문에 범용적이고 통합된 object detection model이다

2. YOLO는 새로운 환경에서 쉽게 적응하기 때문에 빠르고 안정적인 object detection 기능에 의존하는 어플리케이션에 이상적이다.



## Introduction

1. 분류기에 기반한 객체 검출 방법을 사용한 DPM 또는 R-CNN의 동작 방법과는 다르게, YOLO는 회귀 문제로 접근하여 image로부터 bounding box를 바로 찾아내는 방법을 선택했다.

2. 간단한 YOLO pipeline 구조 설명
3. Titan X GPU의 환경에서 초당 45개의 frame을 계산하는 빠른 속도를 자랑한다.
4. YOLO는 siding window나 region proposal-based 과는 다르게 image의 전체 정보를 동시에 처리하기 때문에 image의 feature을 한 번에 파악할 수 있다.
5. 자연 이미지 뿐만 아니라 art image에도 잘 동작함을 확인할 수 있다.

5. 작은 작은 크기의 object에 대해서는 state-of-the-art detection model에 비해서 accuracy는 뒤떨어진다.



## Unified Detection

1. image를 S × S Grid Cell 로 나누고, 각 cell 별로 B 개의 Bounding Box를 만들고, Bounding box에 대한 confidence를 예측한다.

2. Confidence는 아래 표현을 따른다
   $$
   Confidence = Pr(Object) * IOU^{truth}_{pred}
   $$

3.  bounding box는 5개의 predictions를 가진다.

   > - x, y : grid cell 내의 object 중앙 x, y좌표
   > - width, height : 전체 image 대비 width, height 값 
   > - confidence : image 내에 object가 있을 것이라고 확신하는 정도

4. 각 grid cell은 class probability를 예측하는데, 그 표현은 아래와 같다
   $$
   Pr(Class_i|Object)
   $$

5. YOLO Model의 predictions tensor shape은 아래 식을 따른다
   $$
   output: S \times S \times(5*B + C)
   $$

   > **S × S** : number of divided the image(grid cell)
   >
   > **5** :  x, y coordinate, width, height, confidence 
   >
   > **B** : number of Bounding Box
   >
   > **C** : number of Class

6. Pascal VOC dataset에 대해서 YOLO를 사용했고, S = 7, B = 2, C = 20(VOC의 label class 개수) 이므로 final prediction은 7 × 7 × 30 이다.



#### Network Design



![](https://i0.wp.com/thebinarynotes.com/wp-content/uploads/2020/04/Yolo-Architecture.png?fit=678%2C285&ssl=1)

| input image      | 448 × 448 × 3       |                                 |              |                           |
| ---------------- | ------------------- | ------------------------------- | ------------ | ------------------------- |
| Convolution      | kernel size = 7     | num of kernel = 64              | strides = 2  | padding = [kernel size/2] |
| Maxpooling       | kernel size = 2     |                                 | strides = 2  |                           |
| **feature map1** | **112 × 112 × 192** |                                 |              |                           |
| Convolution      | kernel size = 3     | num of kernel = 192             | strides = 1  | padding = [kernel size/2] |
| **feature map2** | **112 × 112 × 64**  |                                 |              |                           |
|                  | **112 × 112 × 256** | **staking feature map 1, 2**    |              |                           |
| Maxpooling       | kernel size = 2     |                                 | strides = 2  |                           |
|                  | **56 × 56 × 256**   |                                 |              |                           |
| Convolution      | kernel size = 1     | num of kernel = 128             | strides = 1  | padding = [kernel size/2] |
| Convolution      | kernel size = 3     | num of kernel = 256             | strides = 1  | padding = [kernel size/2] |
| Convolution      | kernel size = 1     | num of kernel = 256             | strides = 1  | padding = [kernel size/2] |
| Convolution      | kernel size = 3     | num of kernel = 512             | strides = 1  | padding = [kernel size/2] |
| Maxpooling       | kernel size = 2     |                                 | strides = 2  |                           |
|                  | **28 × 28 × 512**   |                                 |              |                           |
| Conv 1           | kernel size = 1     | num of kernel = 256             | strides = 1  | padding = [kernel size/2] |
| Conv 2           | kernel size = 3     | num of kernel = 512             | strides = 1  | padding = [kernel size/2] |
|                  |                     | **4 times iteration conv 1, 2** |              |                           |
| Convolution      | kernel size = 1     | num of kernel = 512             | strides = 1  | padding = [kernel size/2] |
| Convolution      | kernel size = 3     | num of kernel = 1024            | strides = 1  | padding = [kernel size/2] |
| Maxpooling       | kernel size = 2     |                                 | strides = 2  |                           |
|                  | **14 × 14 × 1024**  |                                 |              |                           |
| Conv 3           | kernel size = 1     | num of kernel = 512             | strides = 1  | padding = [kernel size/2] |
| Conv 4           | kernel size = 3     | num of kernel = 1024            | strides = 1  | padding = [kernel size/2] |
|                  |                     | **2 times iteration conv 3, 4** |              |                           |
| Convolution      | kernel size = 3     | num of kernel = 1024            | strides = 1  | padding = [kernel size/2] |
| Convolution      | kernel size = 3     | num of kernel = 1024            | strides = 2  | padding = [kernel size/2] |
|                  | **7 × 7 × 1024**    |                                 |              |                           |
| Convolution      | kernel size = 3     | num of kernel = 1024            | strides = 1  | padding = [kernel size/2] |
| Convolution      | kernel size = 3     | num of kernel = 1024            | strides = 1  | padding = [kernel size/2] |
|                  | **7 × 7 × 1024**    |                                 |              |                           |
| Conn Layer       |                     | Flatten                         |              |                           |
|                  | **50,176 × 1**      |                                 |              |                           |
| Conn Layer       |                     | Dense                           | units = 4096 |                           |
|                  | **4096 × 1**        |                                 |              |                           |
| Conn Layer       |                     | Dense                           | units = 1470 |                           |
|                  | **1470 × 1**        |                                 |              |                           |
|                  |                     | Reshape                         |              |                           |
| **output**       | **7 × 7 × 30**      |                                 |              |                           |



#### Training

1. Darknet이라는 자체 framework로 학습을 진행

2. 작은 size의 object detection을 위해 image size를 224에서 448로 변환했다.

3. 마지막 layer에서 class probability와 bounding box를 predicts

4. bounding box의 위치 정보는 grid cell 기준으로 nomalization을 통해 0~1 사이의 값으로 표현했다.

5. activation function

   마지막 layer에서 linear activation function을 적용했으며, 그 외 layer에서는 **leaky rectified** linear activation function을 사용
   $$
   leaky\ rectified : \ \ \  \O(x) =  \left\{\begin{matrix}
   x, \ \ \ \ \ if\ x>0
   \\ 
   0.1x, \ \ \ \ otherwise
   \end{matrix}\right.
   $$
   
6. loss function

   sum-squared error 를 사용. 간단한 함수이기에 채택했지만 객체 검출에는 알맞지 않음

   object가 없는 grid cell에서는 confidence가 0이 되고, 이러한 confidence가 많아지면 학습이 불안정할 수 있다.

   이를 해결하기 위해 bounding box cofidence predcition 앞에 lambda\_coord 를 곱하고, object가 없는 grid cell의 cofidence predcition 앞에는 lambda\_noodj 를 곱해준다.
   $$
   \lambda_{coord} = 5, \ \ \ \ \ \lambda_{noodj} = 0.5.
   $$
   람다\_coord :  중요도 증가

   람다\_noodj : 중요도 감소

7. sum-squared error에 대한 설명

   object의 크기에 따라서 bounding box의 width, height의 loss 크기가 작더라도, 다른 loss에 비해 상대적으로 큰 차이처럼 영향을 미칠 수 있기 때문에 loss에 루트를 씌운다.

   >  ex) 
   >
   > object 1 의 label width = 300, object 2 의 label width = 16
   >
   > object 1 의 prediction width = 305, object 2 의 prediction width = 13
   >
   > |300 - 305| =5
   >
   > |16 - 13| = 3   
   >
   > 영향은 object 1이 더 작아야 하지만, 값의 크기가  object 2에 비해 크기 때문에 이러한 부분이 학습에 반영되어 의도치 않은 학습 결과를 불러올 수 있다.

8. predicted bounding box 중에서 가장 큰 IOU 값을 가진 bounding box만을 training 과정에서 loss function 비교에 사용하겠다.

9. sum-squared error 수식
   $$
   \lambda_{coord} \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{obj}_{ij}\left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right] \\
   + \lambda_{coord} \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{obj}_{ij}\left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right] \\ 
   + \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{obj}_{ij}(C_i - \hat{C_i})^2\\ 
   + \lambda_{noobj} \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{noobj}_{ij}(C_i - \hat{C_i})^2\\ 
   + \sum^{S^2}_{i = 0}𝟙^{obj}_{i}\sum_{c \in classes} (p_i(c) - \hat{p_i}(c))^2
   $$

   - indicator function

     i : i번째 grid cell

     j : j 번째 detector

     즉, 
     $$
     𝟙^{obj}_{ij}
     $$
     object가 있는 cell에서 2개의 detector 중 j번째 detector가 responsible이면 1
     $$
     𝟙^{obj}_{i}
     $$
     object가 있는 cell일때만 1
     $$
     𝟙^{noobj}_{ij}
     $$
     object가 없는 cell에서 2개의 detector 중 j번째 detector가 responsible이면 1

     

   - localization loss (box 위치 predict)
     $$
     \lambda_{coord} \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{obj}_{ij}\left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right] \\
     + \lambda_{coord} \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{obj}_{ij}\left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right] \\
     $$
     

     > object가 있는 grid cell에서 IOU이 높은 하나의 bounding box에 대해서만  𝟙 이 1의 값을 가지기 때문에, 각 cell당 한 가지 경우에만 loss가 계산된다.

     

   - confidence loss
     $$
     \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{obj}_{ij}(C_i - \hat{C_i})^2\\ 
     + \lambda_{noobj} \sum^{S^2}_{i = 0}\sum^{B}_{j = 0}𝟙^{noobj}_{ij}(C_i - \hat{C_i})^2\\
     $$
     object가 있는 cell의 confidence loss와 없는 cell의 confidence loss를 계산

   - class의 probability loss
     $$
     \sum^{S^2}_{i = 0}𝟙^{obj}_{i}\sum_{c \in classes} (p_i(c) - \hat{p_i}(c))^2
     $$
     object가 있는 cell에서, label class와 predicted class의 probability loss를 계산



10. PASCAL VOC 2007, 2012 data set으로 135번의 epochs동안 학습을 진행했고, batch size는 64, momentum은 0.9, decay는 0.0005로 결정했다.

    learning rate는 처음 75번의 epoch에는 0.001, 그 다음 30번의 epoch에는 0.0001, 다음 30 번의 epoch에는 0.00001로 결정



각가그이 file에 대해서, code를 작성할 때 

논문의 어느 부분을 보고 이렇게 작성했는지에 대한 분석

논문을 올릴 때 code가 함께 올라가는지 질문
