# Day1 code해설
## 목표: Tensorflow 기초 및 Linear Regression Model 구현하기
## 1. Tensorflow 기초(1)
우선 가장 먼저 Tensorflow를 사용하기위해 Tensorflow 라이브러리를 tf로써 import합니다. 
```python
import tensorflow as tf 
```

다음은 tf.constant로 상수를 hello라는 변수에 저장하고 출력합니다.

```python
hello = tf.constant("hello, tensorflow") 
print(hello) 
```

>실행결과
>```python
>Tensor("Const_3:0", shape=(), dtype=string)
>```
>hello가 tensorflow의 `Tensor`라는 자료형이고 `string`을 담고 있음을 알 수 있습니다.

위와 같이 저장된 `Tensor` a,b는 c와 같이 다양한 연산을 수행할 수 있습니다.
```python
a = tf.constant(10)
b = tf.constant(5)
c = tf.add(a,b)
print(c)
```
>실행결과
>```python
>Tensor("Add_3:0", shape=(), dtype=int32)
>```
>여기서 우리는 15가 출력되는 것을 기대 했지만 기대와는 다르게 `Tensor`형태를 출력합니다.   
>그 이유는 Tensorflow 프로그램의 구조가 2가지로 분리되어 있기 떄문입니다.
>>1. 그래프 생성
>>2. 그래프 실행   
>
>**Tensorflow는 텐서와 텐서의 연산들을 미리 정의하여 그래프를 생성하고 이후 필요할때 연산을 실행하는** 
>**코드를 넣어 '원하는시점'에 실제 연산을 수행하도록 합니다.**

그래프의 실행을 위해 `Seesion`을 불러오고 다음과 같이 `run` method를 이용해 실행 합니다.
```python
sess = tf.Session()

print(sess.run(hello))
print(sess.run([a,b,c]))

sess.close()
```
sess의 사용이 끝나면 sess,close()를 이용해 Session을 닫아줍니다.
>실행결과
>```python
>b'hello, tensorflow'
>[10, 5, 15]
>```
---
## 2. Tensorflow 기초(2)

Tensorflow로 프로그래밍시 알아야 할 가장 중요한 개념 중 하나인 `placeholder`는 그래프에 사용할 입력값을 나중에
받기 위해 사용하는 '매개변수'라고 생각하시면 됩니다.

아래와 같이 Placeholder라는 (?,3)모양에 float32 자료형을 가진 텐서를 생성합니다.
```python
X = tf.placeholder(tf.float32,[None,3])
print(X)
```
>실행결과
>```
>Tensor("Placeholder_1:0", shape=(?, 3), dtype=float32)
>```

다음으로 나중에 Placeholder X에 넣을 자료를 생성 합니다.   
단, X가 (?,3)이므로 3개의 요소를 가지게 해줍니다.
```python
x_data = [[1,2,3],[4,5,6]]
```
다음은 `Placeholder`와 같이 중요한 `Variable`을 사용합니다.   
`Variable`은 그래프를 최적화 하는 용도로 학습 함수들이 학습한 결과를 갱신하기 위해 사용됩니다.

아래와 같이 `tf.random_normal` 함수를 이용해 정규분포의 무작위 값으로 변수들을 정의해줍니다.
```python
W = tf.Variable(tf.random_normal([3,2])) //[3,2]행렬형태
b = tf.Variable(tf.random_normal([2,1])) //[2,1]행렬형태
```
>다음처럼 직접 원하는 텐서의 형태의 데이터를 만들어 넣을 수도 있습니다.
>```python
>L = tf.Variable([[0.1,0.2],[0.3,0.4],[0.5,0.6]])
>```

X와 W가 행렬이기 때문에 `tf.matmul`함수를 사용하여 곱해줍니다.
```python
expr = tf.matmul(X,W) + b
```

이제 연산을 실행화고 결과를 출력해 봅니다.
```python
sess = tf.Session()
sess.run(tf.global_variables_initializer()) //앞에서 정의한 변수들 초기화
print("====== x_data========")
print(x_data)
print("====== W =====")
print(sess.run(W))
print("====== b =====")
print(sess.run(b))
print("====== L =====")
print(sess.run(L))
print("====== expr =====")
print(sess.run(expr, feed_dict={X:x_data}))
sess.close()
```
>실행결과
>```python
>====== x_data========
>[[1, 2, 3], [4, 5, 6]]
>====== W =====
>[[-1.2425435   0.5128782 ]
> [-0.41220462  1.3040193 ]
> [ 0.6592981  -0.27851027]]
>====== b =====
>[[-0.4132442]
> [ 1.1030339]]
>====== L =====
>[[0.1 0.2]
> [0.3 0.4]
> [0.5 0.6]]
>====== expr =====
>[[-0.5023026  1.8721418]
> [-1.9723746  8.003582 ]]
>```

## 3. Linear Regression model 구현하기
선형 회기 model의 원리를 모른다면 코드로 구현하기전 <https://pythonkim.tistory.com/10>에서 먼저 원리를 이해하는것을 권합니다.   


X와 Y의 상관관계를 분석하는 기초적인 선형 회귀 모델을 만들기 위해 x,y데이터를 생성합니다.
```python
x_data = [1,2,3]
y_data = [1,2,3]
```
또한 변수 W와 b를 각각 -1.0 ~ 1.0 사이의 균등분포를 가진 무작위 값으로 초기화합니다.
```python
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
```
자료를 입력받을 placeholder를 설정합니다
```python
X = tf.placeholder(tf.float32, name="X") 
//name을 통해 placeholder에 이름을 정해줍니다.
Y = tf.placeholder(tf.float32, name="Y")
```

X와Y의 관계를 분석하기 위한 Linear한 수식을 작성합니다.
```python
hypo = W*X + b
```

다음으로 실제값과 예측값의 차이를 나타내는 손실함수를 작성합니다.
```python
cost = tf.reduce_mean(tf.square(hypo - Y))
```

마지막으로 Tensorflow가 기본 제공하는 `경사하강법` 최적화 함수를 사용해 손실값을 최소화하는
연산 그래프를 생성합니다.
```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train_op = optimizer.minimize(cost)
```

이제 선형 회기 모델을 만들었으니 그래프를 실행하여 학습시키고 결과를 확인해 봅니다.
```python
sess = tf.Session()
sess.run(tf.global_variables_initializer())
    for step in range(100):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})
        print(step,cost_val,sess.run(W),sess.run(b))
``` 
>실행결과
>```python
>0 4.108524 [1.0302837] [0.16469803]
>1 0.051355932 [0.93613964] [0.11964494]
>2 0.0027839812 [0.9478847] [0.12126009]
>3 0.0021006677 [0.9480216] [0.11785419]
>                  .
>                  .
>```
>맨앞의 손실값이 점점 줄어든다면 정상적으로 학습 중입니다.

이제 학습하지 않았던 새로운 값을 넣고 출력을 확인 해 봅니다.
```python
print("X: 5, Y",sess.run(hypo, feed_dict={X: 5}))
print("X: 2.5, Y",sess.run(hypo, feed_dict={X: 2.5}))  
```
>실행결과
>```python
>X: 5, Y [4.9863214]
>X: 2.5, Y [2.4988625]
>```
>처음의 x,y데이터를 [1,2,3]으로 동일하게 하였으니 제대로 학습 되었다면
>y=x의 그래프가 나와야 하며 위 출력을 보면 Y가 X와 거의 비슷한 값을 내는것으로 보아
>정상적으로 학습된것을 확인할수 있습니다.

Day2: [기본신경망 구현하기](https://github.com/wotjd0715/DeepLearning/tree/master/Day2)
