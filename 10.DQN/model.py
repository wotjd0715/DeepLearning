# 알파고를 만든 구글의 딥마인드의 논문을 참고한 DQN 모델을 생성합니다.
# http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html
import tensorflow as tf
import numpy as np
import random
from collections import deque

#지금까지 작성해온 코드와 다르게 DQN은 파이썬 클래스로 작성했습니다. 이는 이해와 수정을 쉽게하기 위함입니다.
class DQN:
    # 학습에 사용할 플레이결과를 얼마나 많이 저장해서 사용할지를 정합니다.
    # (플레이결과 = 게임판의 상태 + 취한 액션 + 리워드 + 종료여부)
    REPLAY_MEMORY = 10000
    # 학습시 사용/계산할 상태값(정확히는 replay memory)의 갯수를 정합니다. (미니배치의 크기입니다.)
    BATCH_SIZE = 32
    # 과거의 상태에 대한 가중치를 줄이는 역할을 합니다.
    GAMMA = 0.99
    # 한 번에 볼 총 프레임 수 입니다.
    # 앞의 상태까지 고려하기 위함입니다.
    STATE_LEN = 4

    # 다음은 DQN객체를 초기화 하는 코드입니다. 텐서플로 세션과 가로와 세로 크기, 행동 개수를 받아 객체를 초기화 합니다.
    def __init__(self, session, width, height, n_action):
        self.session = session
        self.n_action = n_action
        self.width = width
        self.height = height
        # 게임 플레이결과를 저장할 메모리를 만들어주는 코드입니다.
        self.memory = deque()
        # 현재 게임판의 상태
        self.state = None

        # 게임의 상태를 입력받을 변수
        # [게임판의 가로 크기, 게임판의 세로 크기, 게임 상태의 갯수(현재+과거+과거..)]
        self.input_X = tf.placeholder(tf.float32, [None, width, height, self.STATE_LEN])
        # 각각의 상태를 만들어낸 액션의 값들입니다. 0, 1, 2 ..
        self.input_A = tf.placeholder(tf.int64, [None])
        # 손실값을 계산하는데 사용할 입력값입니다. train 함수를 참고하세요.
        self.input_Y = tf.placeholder(tf.float32, [None])

        #다음으로 학습을 진행할 신경망과 목표신경망을 구성합니다. 목표신경망은 단순히 Q값만 구하는데 사용하므로 손실값과 최적화함수를 사용하지 않습니다.
        self.Q = self._build_network('main')
        self.cost, self.train_op = self._build_op()

        # 학습을 더 잘 되게 하기 위해,
        # 손실값 계산을 위해 사용하는 타겟(실측값)의 Q value를 계산하는 네트웍을 따로 만들어서 사용합니다
        self.target_Q = self._build_network('target')

    #_build_network는 앞서 나온 학습 신경망과 목표 신경망을 구성하는 함수입니다. 상태값 input_X를 받아 행동의 가짓수만큼의 출력값을 만듭니다.
    # 이값들의 최댓값을 취해 다음 행동을 정할 것입니다.
    def _build_network(self, name):
        with tf.variable_scope(name):
            model = tf.layers.conv2d(self.input_X, 32, [4, 4], padding='same', activation=tf.nn.relu)
            model = tf.layers.conv2d(model, 64, [2, 2], padding='same', activation=tf.nn.relu)
            model = tf.contrib.layers.flatten(model)
            model = tf.layers.dense(model, 512, activation=tf.nn.relu)

            Q = tf.layers.dense(model, self.n_action, activation=None)

        return Q

    #다음은 DQN의 손실함수를 구하는 부분입니다. 현재 상태를 이용해 학습 신경망으로 구한 Q_value와 다음 상태를 이용해 목표 신경망으로 구한 Q_value(input_Y)를 이용하여 손실값을 구하고 최적화 합니다.
    def _build_op(self):
        # DQN 의 손실 함수를 구성하는 부분입니다. 다음 수식을 참고하세요.
        # Perform a gradient descent step on (y_j-Q(ð_j,a_j;θ))^2
        one_hot = tf.one_hot(self.input_A, self.n_action, 1.0, 0.0)
        Q_value = tf.reduce_sum(tf.multiply(self.Q, one_hot), axis=1)
        #tf.multiply(self.Q, one_hot)함수는 self.Q로 구한 값에서 현재 행동의 인덱스에 해당하는 값만 선택하기 위해 사용합니다.
        cost = tf.reduce_mean(tf.square(self.input_Y - Q_value))
        train_op = tf.train.AdamOptimizer(1e-6).minimize(cost)

        return cost, train_op

    # refer: https://github.com/hunkim/ReinforcementZeroToAll/

    #다음은 목표 신경망을 갱신하는 함수 입니다. 학습 신경망의 변수들의 값을 목표 신경망으로 복사해서 목표 신경망의 변수들을 최신값으로 갱신합니다.
    def update_target_network(self):
        copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        # 학습 네트웍의 변수의 값들을 타겟 네트웍으로 복사해서 타겟 네트웍의 값들을 최신으로 업데이트합니다.
        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))

        self.session.run(copy_op)

    # 현재 상태를 이용해 다음에 최해야 할 행동을 찾는 함수를 만들어 줍니다.
    def get_action(self):
        Q_value = self.session.run(self.Q,
                                   feed_dict={self.input_X: [self.state]})

        action = np.argmax(Q_value[0])

        return action

    # 현재의 상태를 초기화 하는 함수입니다.
    def init_state(self, state):
        # 현재 게임판의 상태를 초기화합니다. 앞의 상태까지 고려한 스택으로 되어 있습니다.
        state = [state for _ in range(self.STATE_LEN)]
        # axis=2 는 input_X 의 값이 다음처럼 마지막 차원으로 쌓아올린 형태로 만들었기 때문입니다.
        # 이렇게 해야 컨볼루션 레이어를 손쉽게 이용할 수 있습니다.
        # self.input_X = tf.placeholder(tf.float32, [None, width, height, self.STATE_LEN])
        self.state = np.stack(state, axis=2)

    #게임 플레이 결과를 받아 메모리에 기억하는 기능을 수행하는 함수입니다.
    def remember(self, state, action, reward, terminal):
        # 학습데이터로 현재의 상태만이 아닌, 과거의 상태까지 고려하여 계산하도록 하였고,
        # 이 모델에서는 과거 3번 + 현재 = 총 4번의 상태를 계산하도록 하였으며,
        # 새로운 상태가 들어왔을 때, 가장 오래된 상태를 제거하고 새로운 상태를 넣습니다.
        next_state = np.reshape(state, (self.width, self.height, 1))
        next_state = np.append(self.state[:, :, 1:], next_state, axis=2)

        # 플레이결과, 즉, 액션으로 얻어진 상태와 보상등을 메모리에 저장합니다.
        self.memory.append((self.state, next_state, action, reward, terminal))

        # 저장할 플레이결과의 갯수를 제한합니다.
        if len(self.memory) > self.REPLAY_MEMORY:
            self.memory.popleft()

        self.state = next_state

    #기억해둔 게임플레이에서 임의의 메모리를 배치크기만큼 가져옵니다. random.sample함수를 통해 임의의 메모리를 가져오고
    #그중 첫 번째 요소를 현재 상탯값으로, 두번재를 다음 상탯값으로, 그리고 취한 행동,보상,게임종료여부를 순서대로 가져온뒤 사용하기 쉽도록 재구성하여 넘겨줍니다.
    def _sample_memory(self):
        sample_memory = random.sample(self.memory, self.BATCH_SIZE)

        state = [memory[0] for memory in sample_memory]
        next_state = [memory[1] for memory in sample_memory]
        action = [memory[2] for memory in sample_memory]
        reward = [memory[3] for memory in sample_memory]
        terminal = [memory[4] for memory in sample_memory]

        return state, next_state, action, reward, terminal

    def train(self):
        # 게임 플레이를 저장한 메모리에서 배치 사이즈만큼을 샘플링하여 가져옵니다.
        state, next_state, action, reward, terminal = self._sample_memory()

        # 학습시 다음 상태를 타겟 네트웍에 넣어 target Q value를 구합니다
        target_Q_value = self.session.run(self.target_Q,
                                          feed_dict={self.input_X: next_state})

        # DQN 의 손실 함수에 사용할 핵심적인 값을 계산하는 부분입니다. 다음 수식을 참고하세요.
        # if episode is terminates at step j+1 then r_j
        # otherwise r_j + γ*max_a'Q(ð_(j+1),a';θ')
        # input_Y 에 들어갈 값들을 계산해서 넣습니다.
        Y = []
        for i in range(self.BATCH_SIZE):
            if terminal[i]:
                Y.append(reward[i])
            else:
                Y.append(reward[i] + self.GAMMA * np.max(target_Q_value[i]))

        self.session.run(self.train_op,
                         feed_dict={
                             self.input_X: state,
                             self.input_A: action,
                             self.input_Y: Y
                         })
