# 게임 구현과 DQN 모델을 이용해 게임을 실행하고 학습을 진행합니다.

#먼저 필요한 모듈을 임포트 합시다. game.py와 model.py를 미리 다운받아 둡시다.
#Game 모듈은 game.py에서 DQN모듈은 model.py에서 받아옵니다.
#Game 모듈은 게임을 진행하고 필요시 matplotlib을 써서 게임상태를 화면에 출력해줍니다.
import tensorflow as tf
import numpy as np
import random
import time

from game import Game
from model import DQN


#에이전트는 학습모드(train)와 게임실행모드(replay)로 나뉩니다. 학습모드일땐 게임화면을 출력하지 않고 빠르게 학습하며 게임실행보모드에선 화면을 출력합니다.
#이릉 위해 에이전트 실행시 모드를 나누어 실행할수 있도록 tf.app.flags를  이용해 실행시 받을 옵션들을 설정합니다.
tf.app.flags.DEFINE_boolean("train", False, "학습모드. 게임을 화면에 보여주지 않습니다.")
FLAGS = tf.app.flags.FLAGS

#다음은 하이퍼 파라미터를 설정해줍시다.
# 최대 학습 횟수
MAX_EPISODE = 10000
# 1000번의 학습마다 한 번씩 타겟 네트웍을 업데이트합니다.
TARGET_UPDATE_INTERVAL = 1000
# 4 프레임마다 한 번씩 학습합니다.
TRAIN_INTERVAL = 4
# 학습 데이터를 어느정도 쌓은 후, 일정 시간 이후에 학습을 시작하도록 합니다. 그전까지는 학습데이터가 적어 학습을 진행해도효과가 크지 않기 때문입니다.
OBSERVE = 100

# action: 0: 좌, 1: 유지, 2: 우 / 게임화면 크기 설정
NUM_ACTION = 3
SCREEN_WIDTH = 6
SCREEN_HEIGHT = 10


def train():
    print('뇌세포 깨우는 중..')
    sess = tf.Session()

    game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, show_game=False) #게임 객체에는 화면크기를 넣어주며 빠른 학습을위해 화면출력은 False로 해줍니다.
    brain = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, NUM_ACTION)
    #DQN객체는 신경망을 학습시키기 위해 Session을 넣어주고 화면을 입력받아 CNN을 구성할 것이므로 화면크기를 넣어 초기설정을 합니다.
    #또한 가장 중요한 신경망의 최종결과값의 개수인 선택할 행동의 갯수(3)을 넣어줍니다.

    #다음으로는 학습 결과를 저장하고 학습상태를 확인하는 코드를 넣어줍니다.
    rewards = tf.placeholder(tf.float32, [None])
    tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('logs', sess.graph)
    summary_merged = tf.summary.merge_all()

    # 타겟 네트웍을 초기화합니다.
    brain.update_target_network()

    # epsilon은 다음에 취할 액션을 DQN 을 이용해 결정할 시기를 결정합니다.
    epsilon = 1.0
    # 프레임 횟수와 학습결과를 확인해기 위해 점수들을 저장할 배열을 초기화 해줍니다.
    time_step = 0
    total_reward_list = []

    # 게임을 시작합니다. MAX_EPISODE만큼 게임을 진행하며 매 게임을 시작하기전에 초기화 해줍니다
    for episode in range(MAX_EPISODE):
        #terminal은 게임의 종료 상태를 나타내며, total_reward는 한게임당 얻은 총 점수입니다.
        terminal = False
        total_reward = 0

        # 게임을 초기화하고 그 상태를 DQN에 초기 상탯값으로 넣어줍니다.
        # 상태는 screen_width x screen_height 크기의 화면 구성입니다.
        state = game.reset()
        brain.init_state(state)

        #부딪치기 전까지 계속 게임을 실행합니다.
        while not terminal:
            # 입실론이 랜덤값보다 작은 경우에는 랜덤한 액션을 선택하고
            # 그 이상일 경우에는 DQN을 이용해 액션을 선택합니다.
            # 초반엔 학습이 적게 되어 있기 때문입니다.
            # 초반에는 거의 대부분 랜덤값을 사용하다가 점점 줄어들어
            # 나중에는 거의 사용하지 않게됩니다.
            if np.random.rand() < epsilon:
                action = random.randrange(NUM_ACTION)
            else:
                action = brain.get_action()

            # 일정 시간이 지난 뒤 부터 입실론 값을 줄입니다.
            # 초반에는 학습이 전혀 안되어 있기 때문입니다.
            if episode > OBSERVE:
                epsilon -= 1 / 1000 # 무작위값을 사용할 비율을 줄이기 위해 입실론 값을 조금씩 줄여줍니다.

            # 결정한 액션을 이용해 게임을 진행하고, 보상과 게임의 종료 여부를 받아옵니다.
            state, reward, terminal = game.step(action)
            total_reward += reward

            # 현재 상태를 Brain에 기억시킵니다.
            # 기억한 상태를 이용해 학습하고, 다음 상태에서 취할 행동을 결정합니다.
            brain.remember(state, action, reward, terminal)

            #현재 프레임이 100번(OBSERVE)이 넘으면 4프레임(TRAIN_INTERVAL)마다 한번식 학습을 진행합니다.
            if time_step > OBSERVE  and time_step % TRAIN_INTERVAL == 0:
                # DQN 으로 학습을 진행합니다.
                brain.train()

            #또한 1000프레임(TARGET_UPDATE_INTERVAL)마다 한번씩 목표 신경망을 갱신해줍니다.
            if time_step % TARGET_UPDATE_INTERVAL == 0:
                # 타겟 네트웍을 업데이트 해 줍니다.
                brain.update_target_network()

            time_step += 1

        #게임이 종료되면 획득한 점수를 출력하고 이번 에피소드에서 받은 점수를 저장합니다. 그리고 에피소드 10번마다 받은점수 log에 저장하고, 100번마다 학습된 모델을 저장합니다.
        print('게임횟수: %d 점수: %d' % (episode + 1, total_reward))

        total_reward_list.append(total_reward)

        if episode % 10 == 0:
            summary = sess.run(summary_merged, feed_dict={rewards: total_reward_list})
            writer.add_summary(summary, time_step)
            total_reward_list = []

        if episode % 100 == 0:
            saver.save(sess, 'model/dqn.ckpt', global_step=time_step)

#이제 학습결과를 실행하는 함수를 작성해줍니다.
#결과를 실행하는 replay()함수는 train함수에서 학습부분만 빠진것과 비슷합니다.
#한가지 주의할점은 텐서플로 세션을 새로 생성하지 않고 tf.train.Saver()로 저장해둔 모델을 읽어와서 생성해야 한다는 것입니다.
def replay():
    print('뇌세포 깨우는 중..')
    sess = tf.Session()

    game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, show_game=True)
    brain = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, NUM_ACTION)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('model')
    saver.restore(sess, ckpt.model_checkpoint_path)

    # 게임을 시작합니다.
    for episode in range(MAX_EPISODE):
        terminal = False
        total_reward = 0

        state = game.reset()
        brain.init_state(state)

        while not terminal:
            action = brain.get_action()

            # 결정한 액션을 이용해 게임을 진행하고, 보상과 게임의 종료 여부를 받아옵니다.
            state, reward, terminal = game.step(action)
            total_reward += reward

            brain.remember(state, action, reward, terminal)

            # 게임 진행되는 속도가 인간이 인지할 수 있는 속도로 보여줍니다.
            time.sleep(0.3)

        print('게임횟수: %d 점수: %d' % (episode + 1, total_reward))

# 마지막으로 애아전트를 학습용으로 실행할지 replay로 실행할지 선택할수 있게 하는 부분을 만들어 줍니다.
def main(_):
    if FLAGS.train:
        train()
    else:
        replay()


if __name__ == '__main__':
    tf.app.run()
#logs와 model 폴더는 미리 만들어 둡니다.
# python agent.py --train 을 입력할경우 학습이 진행되며
# python agent.py 를 입력시 게임이 실행됩니다.