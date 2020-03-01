# Day5 CNN
## 목표: CNN을 이용하여 MNIST데이터셋을 학습시켜 보자.
### **1. CNN란?**
![git](https://github.com/wotjd0715/DeepLearning/blob/master/Day5/cnn1.png)   
CNN 모델은 기본적으로 위 그림과 같이 ```convolution```계층과 ```pooling```계층으로 구성됩니다. 간단히 설명 해보자면 
컨볼루션 계층은 이미지를 일정 부분으로 나누어 가중치를 적용하여 하나의 값으로 만들어 나타내고 그 뒤 풀링계층을 이용해 다시 각 영역의
대표값으로 대체해 특징지도의 크기를 축소 시킵니다. 이로 인해 기존의 신경망보다 계산량이 매우 적어 학습이 더 빠르고 효율적으로 이루어 집니다.
자세한 설명은 아래 링크로 대체하겠습니다.   
<http://taewan.kim/post/cnn/>
