
import gym
import matplotlib.pyplot as plt

import numpy as np

from keras import layers
from keras.models import Model
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input
from keras.initializers import glorot_uniform
from keras.layers import advanced_activations
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers


def choose_action(PGNN, _observation, i_episode):
    if i_episode == 0:
        _action = np.random.choice(range(2))
    else:

        #observation = np.reshape(observation, (1, 4))
        # observation = np.array(observation).astype('float64')[:-1, :]

        _observation = np.array(_observation).astype('float64')
        # print('_observation:  ', _observation, 'np.shape(_observation):  ', np.shape(_observation))
        # x = observation[:-1, :]
        a_prob = PGNN.predict(np.reshape(_observation, (1, 4)))
        # print('a_prob:  ', a_prob)

        # action = np.random.choice(range(a_prob.shape[1]), p=a_prob.ravel())
        _action = np.random.choice(range(2), p=a_prob.ravel())
        # print('_action:  ', _action)
    return _action


def store_transition(s, a, r, ep_obs, ep_as, ep_rs, n):
    a_onehot = np_utils.to_categorical(a, num_classes=n)

    # print('s: ', s)
    # print('a: ', a)
    # print('r: ', r)
    ep_obs.append(s)
    ep_as.append(a_onehot)
    ep_rs.append(r)
    return ep_obs, ep_as, ep_rs


def _custom_loss(y_true, y_pred):
    # loss = -1 * K.mean(y_true * K.log(y_pred))
    '''
    print('y_true:  ', y_true)
    print('y_pred:  ', y_pred)
    print('K.log(y_pred):  ', K.log(y_pred))
    print('y_true * K.log(y_pred):  ', y_true * K.log(y_pred))
    '''
    loss = -1 * K.mean(y_true * K.log(y_pred))
    # print('loss:  ', K.eval(loss))

    return loss


def build_net(n_features, n_actions):
    # print('n_features:  ', n_features)
    input_ = Input(shape=[n_features])
    den_1 = Dense(32, activation='relu')(input_)
    # den_1 = BatchNormalization()(den_1)
    all_act = Dense(n_actions)(den_1)  #
    all_act_prob = Dense(n_actions, activation='softmax')(all_act)
    model = Model(inputs=input_, outputs=all_act_prob)

    optimizer = optimizers.rmsprop(
        # lr=0.001,
        # clipnorm=0.8,
        # clipvalue=0.8
    )

    optimizer_ = optimizers.adam(
        lr=0.001,
        # clipnorm=0.8
    )
    model.compile(
        optimizer=optimizer,
        loss=_custom_loss
    )
    model.summary()
    return model


def discount_and_norm_rewards(ep_rs, gamma):
    # discount episode rewards
    discounted_ep_rs = np.zeros_like(ep_rs)
    running_add = 0
    for t in reversed(range(0, len(ep_rs))):
        running_add = running_add * gamma + ep_rs[t]
        discounted_ep_rs[t] = running_add

    # normalize episode rewards
    discounted_ep_rs -= np.mean(discounted_ep_rs)
    discounted_ep_rs /= np.std(discounted_ep_rs)
    return discounted_ep_rs


def learn(PGNN, ep_obs, ep_as, ep_rs):
    discounted_ep_rs_norm = discount_and_norm_rewards(ep_rs, gamma=0.99)
    discounted_ep_rs_norm = np.reshape(discounted_ep_rs_norm, (len(discounted_ep_rs_norm), 1))
    print('--', np.shape(ep_as), np.shape(ep_rs), np.shape(discounted_ep_rs_norm))

    # advantages
    fake_labels = ep_as * discounted_ep_rs_norm

    '''
    PGNN.fit(
        ep_obs, ep_as,
        epochs=1,
        # batch_size=np.shape(ep_obs)[0]
    )
    '''
    loss = PGNN.train_on_batch(ep_obs, fake_labels)
    print('loss:  ', loss, np.shape(ep_obs))

    return discounted_ep_rs_norm


def pgnn_init(env, PG, load):
    ver = 'v6'
    print('PGNN Version :  ', ver)

    n_actions = env.action_space.n
    n_features = env.observation_space.shape[0]
    PGNN = build_net(n_actions=n_actions, n_features=n_features)

    if load == True:
        filepath='./model/pgnn_' + str(ver) + '.h5'     # if load == True
        PGNN.load_weights(filepath)
    return PGNN, ver


if __name__ == '__main__':
    # RENDER = False  # 在屏幕上显示模拟窗口会拖慢运行速度, 我们等计算机学得差不多了再显示模拟
    # DISPLAY_REWARD_THRESHOLD = 400  # 当 回合总 reward 大于 400 时显示模拟窗口
    MAX_EP = 5000
    MAX_T = 2000

    env = gym.make('CartPole-v0')  # CartPole 这个模拟
    env = env.unwrapped  # 取消限制

    # very important parameter
    env.seed(1)  # 普通的 Policy gradient 方法, 使得回合的 variance 比较大, 所以我们选了一个好点的随机种子

    goal_average_steps = 300

    # 只存储最近num_consecutive_iterations场的得分（可以理解为是一个容量为num_consecutive_iterations的栈）
    num_consecutive_iterations = 50
    ep_score = np.zeros(num_consecutive_iterations)

    running_best = 0

    time = 0

    ep_obs = []
    ep_as = []
    ep_rs = []

    rs_his = []
    score_his = []

    print('----------------------------------------------------------')
    print(env.action_space)  # 显示可用 action
    print(env.action_space.n)
    print(env.observation_space)  # 显示可用 state 的 observation
    print(env.observation_space.shape[0])
    print(env.observation_space.high)  # 显示 observation 最高值
    print(env.observation_space.low)  # 显示 observation 最低值
    print('----------------------------------------------------------')

    # ver, rs_his, score_his = play_from_pgnn(env, PG, ep_obs, ep_as, ep_rs, running_best, goal_average_steps)

    #######################################################################
    PGNN, ver = pgnn_init(
        env,
        PG=None,
        load = False,
    )
    #######################################################################

    done = False

    for i_episode in range(MAX_EP):
        print('\n\nagent is playing ' + str(i_episode) + ' games')
        observation = env.reset()

        # observation = np.reshape(observation, (4,))
        observation = np.reshape(observation, [1, env.observation_space.shape[0]])
        # observation = Standardize(observation, -1, 1)
        # print('observation:  ', observation)

        while True:
        # for time in range(MAX_T):
            time += 1
            ## if RENDER: env.render()
            env.render()
            # print('np.shape(ep_obs):  ', np.shape(ep_obs))

            #######################################################################
            action = choose_action(PGNN, observation, i_episode)
            #######################################################################

            # print('action:  ', action)
            observation_, reward, done, info = env.step(action)

            #######################################################################
            ep_obs, ep_as, ep_rs = store_transition(observation_, action, reward, ep_obs, ep_as, ep_rs,
                                                       env.action_space.n)  # 存储这一回合的 transition
            #######################################################################

            if done or time > MAX_T :

                #######################################################################
                ep_rs_sum = sum(ep_rs)
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                ## if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # 判断是否显示模拟
                print("episode:", i_episode, "  reward:", int(running_reward))
                rs_his.append([running_reward])

                if running_reward > running_best:
                    print('running_best: ', running_best)
                    print('running_reward: ', running_reward)
                    running_best = running_reward

                    # PGNN.save_weights('./model/best_try_PG_' + str(ver) + '.h5')
                    # print('model saved!')

                ep_obs = np.array(ep_obs).astype('float64')
                # x = ep_obs[:-1, :]

                ep_as = np.array(ep_as).astype('int')
                # y = ep_as[:-1, :]

                # vt = PG.learn(PGNN, x, y, ep_rs)  # 学习, 输出 vt, 我们下节课讲这个 vt 的作用
                vt = learn(PGNN, ep_obs, ep_as, ep_rs)

                ep_obs = []
                ep_rs = []
                ep_as = []

                if i_episode == 0:
                    plt.plot(vt)  # plot 这个回合的 vt
                    plt.xlabel('episode steps')
                    plt.ylabel('normalized state-action value')
                    plt.show()

                #######################################################################

                ep_score = np.hstack((ep_score[1:], [time]))  # 更新最近10场游戏的得分stack
                print('last_time_steps:\n', ep_score)
                print("episode: {}/{}, score: {}".format(i_episode, MAX_EP, time))
                score_his.append(time)

                break

            observation = observation_

        if (ep_score.mean() >= goal_average_steps):
            #######################################################################
            # PGNN.save('./model/pgnn_' + str(ver) + '.h5')
            #######################################################################

            print('model saved!')
            print('ep_score_avg:  ', ep_score.mean())
        else:
            print('model save jumped!')
            # break

    # np.savetxt('./info/pg_rs_his_best_try_' + str(ver) + '.csv', rs_his)
    # np.savetxt('./info/pg_score_his_best_try_' + str(ver) + '.csv', score_his)

    # print(rs_his)
