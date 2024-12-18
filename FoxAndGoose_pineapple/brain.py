import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(1)
tf.random.set_seed(1)

class DeepQNetwork(tf.keras.Model):
    def __init__(self, n_actions_fox, n_actions_goose, n_features, number_goose, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=300, memory_size=500, batch_size=32, e_greedy_increment=None):
        super(DeepQNetwork, self).__init__()

        # 环境参数
        self.n_actions_fox = n_actions_fox  # 16 actions for the fox
        self.n_actions_goose = n_actions_goose  # 鹅的动作空间大小
        self.n_features = n_features  # 输入特征的形状（现在是一个 (7, 7) 的矩阵）
        self.lr = learning_rate
        self.gamma = reward_decay  # 奖励衰减因子
        self.epsilon_max = e_greedy  # 最大 epsilon
        self.replace_target_iter = replace_target_iter  # 更新 target 网络的步数
        self.memory_size = memory_size  # 经验池大小
        self.batch_size = batch_size  # 批量大小
        self.epsilon_increment = e_greedy_increment  # epsilon 增量
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max  # 初始 epsilon
        self.number_goose = number_goose

        # 存储经验
        self.memory = np.zeros((self.memory_size, np.prod(n_features) * 2 + 2))  # 经验池
        self.learn_step_counter = 0  # 学习步数

        # 构建 Q 网络
        self.eval_net = self._build_net()
        self.target_net = self._build_net()
        self.target_net.set_weights(self.eval_net.get_weights())  # 初始化时将 eval_net 的权重赋值给 target_net
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr)

    def _build_net(self):
        # 构建评估网络（Q 网络）
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=self.n_features),  # 输入层，形状为 (n_features,)
            tf.keras.layers.Flatten(),  # 展平层
            tf.keras.layers.Dense(64, activation='relu'),  # 隐藏层
            tf.keras.layers.Dense(self.n_actions_fox + self.n_actions_goose)  # 输出层，返回狐狸和鹅的所有动作 Q 值
        ])
        return model

    def store_transition(self, s, a, r, s_):
        """存储经验"""
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s.flatten(), [a, r], s_.flatten()))  # 将状态展平
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, role):
        """选择动作，使用ε-greedy策略"""
        # 确保输入的观察是正确的形状
        observation = np.expand_dims(observation, axis=0)  # 添加 batch 维度，变成 (1, 7, 7)

        # 使用 epsilon-greedy 策略
        if np.random.uniform() < self.epsilon:

            if role == "fox":
                fox_actions_value = self.eval_net(observation)  # 评估网络输出所有动作的 Q 值
                action = np.argmax(fox_actions_value[:, :self.n_actions_fox])  # 选择狐狸的动作
            elif role == "goose":

                # 获取所有鹅的动作 Q 值
                goose_actions_value = self.eval_net(observation)[:, self.n_actions_goose:]

                # 计算每只鹅的 Q 值，选择最大 Q 值对应的鹅
                goose_q_values = np.max(goose_actions_value, axis=1)  # 获取每只鹅的最大 Q 值
                best_goose_id = np.argmax(goose_q_values)  # 找到 Q 值最大的鹅（即最佳鹅）
                # 根据找到的最佳鹅选择动作
                action = np.argmax(goose_actions_value[0, best_goose_id]), best_goose_id  # 选择鹅的动作，并返回鹅编号

        else:
            if role == "fox":
                action = np.random.randint(0, self.n_actions_fox)  # 随机选择狐狸的动作
            elif role == "goose":
                print("????")
                goose_index = np.random.randint(0, self.number_goose)  # 随机选择鹅的编号
                action = np.random.randint(0, self.n_actions_goose), goose_index  # 随机选择动作并返回鹅的编号

        return action

    def learn(self):
        """学习阶段"""
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.set_weights(self.eval_net.get_weights())  # 更新 target 网络的权重
            print('\nTarget network updated.\n')

        # 从经验池中随机抽取一个批次
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 计算目标 Q 值
        q_next = self.target_net(batch_memory[:, -self.n_features:]).numpy()
        q_eval = self.eval_net(batch_memory[:, :self.n_features]).numpy()

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # 计算损失并应用梯度更新
        with tf.GradientTape() as tape:
            q_values = self.eval_net(batch_memory[:, :self.n_features])
            loss = tf.reduce_mean(tf.square(q_target - q_values))

        gradients = tape.gradient(loss, self.eval_net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.eval_net.trainable_variables))

        # 更新 epsilon
        if self.epsilon_increment is not None:
            self.epsilon = min(self.epsilon + self.epsilon_increment, self.epsilon_max)

        self.learn_step_counter += 1
        return loss

    def plot_cost(self):
        """绘制学习过程中损失变化图"""
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
