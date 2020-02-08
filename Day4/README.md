# Day4 MNIST
## 목표: MNIST데이터셋을 신경망으로 학습시켜 보자.
### **1. MNIST란?**
MNIST란 손글씨로 쓴 숫자들의 이미지를 모아놓은 데이터셋으로 **0~9**까지의
숫자를 **29x29**픽셀의 크기의 이미지로 구성해 놓은 것입니다.   
+ [다른 MNIST 데이터셋 다운받기](http://yann.lecun.com/exdb/mnist/)


### **2. Overfitting이란?**
머신러닝을 학습시킬때 우리가 가지고 있는 데이터가 100개라고 한다면 "100개를
모두 학습시키는데 사용하는 것"(**학습데이터**로 사용하는 것)이 가장 잘 학습 시킬것
같지만 그렇게 학습시킬 경우 우리가 "학습시킨 데이터에 대한 error는 낮지만 실제 데이터에
대한 오차률은 증가하게 됩니다"(**Overfitting**이 발생 합니다).   
즉, 컴퓨터는 인간과 다르게 응용력이 없어 네발 자전거를 타면 탈수록 네발 자전거는
더욱 잘 타게 되지만 거기에 완전히 익숙해져 버려 비슷하지만 조금 차이가 있는 두발 자전거는
탈 수 없게 됩니다.  
따라서 우리는 100개의 데이터가 있으면 80개 정도는 **학습데이터**(네발 자전거)로 사용하고
나머지 20개는 **테스트 데이터**(두발 자전거)로 사용하여 Overfitting이 되는지를 확인하는데
사용합니다.

![git](https://github.com/wotjd0715/DeepLearning/blob/master/Day4/over.png)

### **3. DropOut**
Dropout은 위에서 말한 Overfitting을 방지하기 위해 나온 개념으로 학습 단계마다 일부분의 뉴런을 제거하여
학습데이터의 특징들이 특정 뉴런들에 고정되는 것을 막아 가중치의 균형을 잡도록 한다.
다만, 일부 뉴런을 학습에서 제외하므로 충분히 학습하는데 시간이 조금 더 오래 걸린다.  
 
![git](https://github.com/wotjd0715/DeepLearning/blob/master/Day4/dropout.png)   

**p.s** Overfitting을 막는 다른 방법중 ```Batch Normalization```기법이 있는데 이는 학습속도 역시 
향상시켜 주는 장점이 있다.

***
##### 사진출처
https://untitledtblog.tistory.com/68 (Overfitting)
https://pythonkim.tistory.com/42 (Dropout)
