# Day8 RNN
##  목표: 자연어 처리나 음성인식에 강점을 가진 RNN모델을 공부해 보자

### **1. RNN이란?**
RNN은 Recurrent Neural Network의 약자로 우리나라 말로는 순환신경망이라는 이름을 가지고 있습니다.   

![git](https://github.com/wotjd0715/DeepLearning/blob/master/Day8/RNN1.png)   
RNN의 개념은 다른 신경망과는 조금 다릅니다.  
우선 위 그림의 가운데에 있는 하나의 덩어리의 신경망을 ``Cell``이라 하며 RNN은 이 셀을 여러개로 중첩하여 만듭니다.   
즉, 앞에서 학습한 결과를 다음 학습의 입력으로 사용하는것인데, 이런 구조로 인하여 **학습 데이터를 단계별로 구분하여 입력해야 합니다.**(위 그림에선 X1,X2..로 구분)   

### **2. Sequence to Sequence란?**
줄여서 ``Seq2Seq``이라 부르는 이 모델은 구글이 기게번역에 사용하는 신경망 모델입니다.   
순차적으로 정보를 받는 ``RNN``과 출력하는 신경망을 조합한 모델로, 번역 등 문장을 입력받아
다른 문장을 출력하는 프로그램에서 많이 사용합니다. 

![git](https://github.com/wotjd0715/DeepLearning/blob/master/Day8/s2s.png)

인코더에서는 원문을 디코더에서는 번역한 결과물을 입력 받습니다. 그후, 디코더가 출력한 결과물을 변역된 결과물과 비교해가면서
학습합니다.    
Seq2Seq 모델에서는 위 그림처럼 특수한 ``Symbol``이 필요한데   
디코더에 입력의 시작을 알려주는 심볼(bos), 디코더 입력의 종료를 알려주는 심볼(eos), 빈 데이터를
채울대 사용하는 아무 의미없는 심볼입니다.   
   
      
##### **사진출처**   
https://pythonkim.tistory.com/57 (RNN)
https://d2l.ai/chapter_recurrent-modern/seq2seq.html (Seq2Seq)
