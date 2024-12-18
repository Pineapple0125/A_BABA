import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


# 定义 DQN 模型
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # 全连接层 1
        self.fc2 = nn.Linear(128, 128)  # 全连接层 2
        self.fc3 = nn.Linear(128, output_size)  # 输出层，计算 Q 值

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用 ReLU 激活函数
        x = torch.relu(self.fc2(x))  # 使用 ReLU 激活函数
        x = self.fc3(x)  # 输出 Q 值
        return x


class Fox:
    def __init__(self, fox_x, fox_y):
        self.fox_x = fox_x
        self.fox_y = fox_y


class Goose:
    def __init__(self, goose_x, goose_y, goose_index):
        self.goose_x = goose_x
        self.goose_y = goose_y
        self.index = goose_index


# 玩家类
class Player:
    def __init__(self, is_auto=False, is_fox=False):
        self.is_auto = is_auto  # 是否自动操作
        self.is_fox = is_fox  # 是否是狐狸
        self.agent = DQN(input_size=49, output_size=49)  # 输入为 7x7 扁平化棋盘，输出为 49 个可能的动作
        self.optimizer = optim.Adam(self.agent.parameters(), lr=0.001)  # Adam 优化器
        self.criterion = nn.MSELoss()  # MSE 损失函数
        self.replay_buffer = deque(maxlen=1000)  # 经验回放缓冲区
        self.epsilon = 1  # 探索率
        self.epsilon_min = 0.1  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.gamma = 0.99  # 折扣因子
        self.round = 0
        self.fox_q_table = None
        self.goose_q_table = None

        self.goose_number = 13
        self.fox_number = 1

        self.fox = Fox(3, 3)
        self.goose_list = [Goose(goose_x=x, goose_y=y, goose_index=index)
                           for index, (x, y) in enumerate(
                [(3, 0), (3, 6), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
                 (5, 2), (5, 3), (5, 4), (6, 2), (6, 3), (6, 4)]
            )]

        self.state = [
            [' ', ' ', '.', '.', '.', ' ', ' '],  # 0  1  2  3  4  5  6
            [' ', ' ', '.', '.', '.', ' ', ' '],  # 7  8  9 10 11 12 13
            ['.', '.', '.', '.', '.', '.', '.'],  # 14 15 16 17 18 19 20
            ['G', '.', '.', 'F', '.', '.', 'G'],  # 21 22 23 24 25 26 27
            ['G', 'G', 'G', 'G', 'G', 'G', 'G'],  # 28 29 30 31 32 33 34
            [' ', ' ', 'G', 'G', 'G', ' ', ' '],  # 35 36 37 38 39 40 41
            [' ', ' ', 'G', 'G', 'G', ' ', ' ']  # 42 43 44 45 46 47 48
        ]

        self.state_last = []

        self.state_mapping = {
            ' ': 0,  # 空格
            '.': 1,  # 小点
            'G': 2,  # 鹅
            'F': 3  # 狐狸
        }
        self.state_space = self.convert_state_to_numbers(self.state)

        self.fox_action_list = [
            (0, 1, 0),  # → not kill
            (1, 1, 0),  # ↘
            (1, 0, 0),  # ↓
            (1, -1, 0),  # ↙
            (0, -1, 0),  # ←
            (-1, -1, 0),  # ↖
            (-1, 0, 0),  # ↑
            (-1, 1, 0),  # ↗

            (0, 1, 1),  # → kill
            (1, 1, 1),  # ↘
            (1, 0, 1),  # ↓
            (1, -1, 1),  # ↙
            (0, -1, 1),  # ←
            (-1, -1, 1),  # ↖
            (-1, 0, 1),  # ↑
            (-1, 1, 1)  # ↗
        ]


        self.goose_action_list = [
            (0, 1, 0),  # → not kill
            (1, 1, 0),  # ↘
            (1, 0, 0),  # ↓
            (1, -1, 0),  # ↙
            (0, -1, 0),  # ←
            (-1, -1, 0),  # ↖
            (-1, 0, 0),  # ↑
            (-1, 1, 0)  # ↗
        ]

        self._fox_base_multi = [1, 2, 4, 6]

        self.fox_rule = {
            #            →  ↘ ↓  ↙ ←  ↖  ↑  ↗
            2: self.mix([2, 3, 4, 0, 0, 0, 0, 0]),
            3: self.mix([1, 0, 1, 0, 1, 0, 0, 0]),
            4: self.mix([0, 0, 4, 3, 2, 0, 0, 0]),
            9: self.mix([2, 0, 3, 0, 0, 0, 1, 0]),
            #             →  ↘ ↓  ↙ ←  ↖  ↑  ↗
            10: self.mix([1, 2, 3, 2, 1, 1, 1, 1]),
            11: self.mix([0, 0, 3, 0, 2, 0, 1, 0]),
            14: self.mix([4, 3, 2, 0, 0, 0, 0, 0]),
            15: self.mix([3, 0, 2, 0, 1, 0, 0, 0]),
            16: self.mix([3, 2, 3, 2, 2, 0, 2, 2]),
            17: self.mix([2, 0, 3, 0, 2, 0, 2, 0]),
            18: self.mix([2, 2, 3, 2, 3, 2, 2, 0]),
            19: self.mix([1, 0, 2, 0, 3, 0, 0, 0]),
            20: self.mix([0, 0, 2, 3, 4, 0, 0, 0]),
            21: self.mix([4, 0, 1, 0, 0, 0, 1, 0]),
            22: self.mix([3, 2, 1, 1, 1, 1, 1, 2]),
            23: self.mix([3, 0, 2, 0, 2, 0, 2, 0]),
            24: self.mix([2, 1, 2, 1, 2, 1, 2, 1]),
            25: self.mix([2, 0, 2, 0, 3, 0, 2, 0]),
            26: self.mix([1, 1, 1, 2, 3, 2, 1, 1]),
            27: self.mix([0, 0, 1, 0, 4, 0, 1, 0]),
            28: self.mix([4, 0, 0, 0, 0, 0, 2, 3]),
            29: self.mix([3, 0, 0, 0, 1, 0, 2, 0]),
            30: self.mix([3, 2, 2, 0, 2, 2, 3, 2]),
            31: self.mix([2, 0, 2, 0, 2, 0, 3, 0]),
            32: self.mix([2, 0, 2, 2, 3, 2, 3, 2]),
            33: self.mix([1, 0, 0, 0, 3, 0, 2, 0]),
            34: self.mix([0, 0, 0, 0, 4, 3, 2, 0]),
            37: self.mix([2, 0, 1, 0, 0, 0, 3, 0]),
            38: self.mix([1, 1, 1, 1, 1, 2, 3, 2]),
            39: self.mix([0, 0, 1, 0, 2, 0, 3, 0]),
            44: self.mix([2, 0, 0, 0, 0, 0, 4, 3]),
            45: self.mix([1, 0, 0, 0, 1, 0, 4, 0]),
            46: self.mix([0, 0, 0, 0, 2, 3, 4, 0]),
        }

        self.goose_rule = {
            #            →  ↘ ↓  ↙ ←  ↖  ↑  ↗
            2: self.mix([1, 1, 1, 0, 0, 0, 0, 0]),
            3: self.mix([1, 0, 1, 0, 1, 0, 0, 0]),
            4: self.mix([0, 0, 1, 1, 1, 0, 0, 0]),
            9: self.mix([1, 0, 1, 0, 0, 0, 1, 0]),
            #             →  ↘ ↓  ↙ ←  ↖  ↑  ↗
            10: self.mix([1, 1, 1, 1, 1, 1, 1, 1]),
            11: self.mix([0, 0, 1, 0, 1, 0, 1, 0]),
            14: self.mix([1, 1, 1, 0, 0, 0, 0, 0]),
            15: self.mix([1, 0, 1, 0, 1, 0, 0, 0]),
            16: self.mix([1, 1, 1, 1, 1, 0, 1, 1]),
            17: self.mix([1, 0, 1, 0, 1, 0, 1, 0]),
            18: self.mix([1, 1, 1, 1, 1, 1, 1, 0]),
            19: self.mix([1, 0, 1, 0, 1, 0, 0, 0]),
            20: self.mix([0, 0, 1, 1, 1, 0, 0, 0]),
            21: self.mix([1, 0, 1, 0, 0, 0, 1, 0]),
            22: self.mix([1, 1, 1, 1, 1, 1, 1, 1]),
            23: self.mix([1, 0, 1, 0, 1, 0, 1, 0]),
            24: self.mix([1, 1, 1, 1, 1, 1, 1, 1]),
            25: self.mix([1, 0, 1, 0, 1, 0, 1, 0]),
            26: self.mix([1, 1, 1, 1, 1, 1, 1, 1]),
            27: self.mix([0, 0, 1, 0, 1, 0, 1, 0]),
            28: self.mix([1, 0, 0, 0, 0, 0, 1, 1]),
            29: self.mix([1, 0, 0, 0, 1, 0, 1, 0]),
            30: self.mix([1, 1, 1, 0, 1, 1, 1, 1]),
            31: self.mix([1, 0, 1, 0, 1, 0, 1, 0]),
            32: self.mix([1, 0, 1, 1, 1, 1, 1, 1]),
            33: self.mix([1, 0, 0, 0, 1, 0, 1, 0]),
            34: self.mix([0, 0, 0, 0, 1, 1, 1, 0]),
            37: self.mix([1, 0, 1, 0, 0, 0, 1, 0]),
            38: self.mix([1, 1, 1, 1, 1, 1, 1, 1]),
            39: self.mix([0, 0, 1, 0, 1, 0, 1, 0]),
            44: self.mix([1, 0, 0, 0, 0, 0, 1, 1]),
            45: self.mix([1, 0, 0, 0, 1, 0, 1, 0]),
            46: self.mix([0, 0, 0, 0, 1, 1, 1, 0]),
        }

    def mix(self, multi_list):
        result = []
        for i in range(8):
            for j in range(multi_list[i]):
                result.append(
                    (
                    self.fox_action_list[i][0] * self._fox_base_multi[j], self.fox_action_list[i][1] * self._fox_base_multi[j]))
        return result

    def convert_state_to_numbers(self, state_grid):
        numerical_state_str = ''.join(''.join(str(self.state_mapping[cell]) for cell in row) for row in state_grid)
        return numerical_state_str

    def convert_number_to_state(self, numerical_state_str):
        # 将数字字符串分割成对应的数字，并恢复成二维网格
        numerical_state_list = [int(char) for char in numerical_state_str]
        inverse_state_mapping = {v: k for k, v in self.state_mapping.items()}
        restored_state = []
        for i in range(7):
            restored_state.append(
                [inverse_state_mapping.get(num, ' ') for num in numerical_state_list[i * 7:(i + 1) * 7]])

        return restored_state

    def build_Q_table(self):
        init_fox_data = {
            "state": ["0011100001110011111112113112222222200222000022200"],
            "(0, 1, 0)": [0], "(1, 1, 0)": [0], "(1, 0, 0)": [0],
            "(1, -1, 0)": [0], "(0, -1, 0)": [0], "(-1, -1, 0)": [0],
            "(-1, 0, 0)": [0], "(-1, 1, 0)": [0], "(0, 1, 1)": [0],
            "(1, 1, 1)": [0], "(1, 0, 1)": [0], "(1, -1, 1)": [0],
            "(0, -1, 1)": [0], "(-1, -1, 1)": [0], "(-1, 0, 1)": [0],
            "(-1, 1, 1)": [0]
        }
        self.fox_q_table = pd.DataFrame(init_fox_data, columns=["state"] + self.fox_action_list)

        init_goose_data = {
            "state": ["0011100001110011111112113112222222200222000022200"],
            "(0, 1, 0)": [0], "(1, 1, 0)": [0], "(1, 0, 0)": [0],
            "(1, -1, 0)": [0], "(0, -1, 0)": [0], "(-1, -1, 0)": [0],
            "(-1, 0, 0)": [0], "(-1, 1, 0)": [0]
        }
        self.goose_q_table = pd.DataFrame(init_goose_data, columns=["state"] + self.goose_action_list)

        print(self.goose_q_table)
        print(self.fox_q_table)

    # 选择一个动作（使用 epsilon-greedy 策略）
    def goose_select_action(self, state, q_table):
        state_action = q_table.loc[state, :8]
        if (np.random.uniform() > self.epsilon) or (state_action.all() == 0):
            action_name = np.random.choice(self.goose_action_list)
        else:
            action_name = state_action.argmax()
        return action_name

    def fox_select_action(self, state, q_table):
        state_action = q_table.loc[state, :]
        if (np.random.uniform() > self.epsilon) or (state_action.all() == 0):
            action_name = np.random.choice(self.fox_action_list)
        else:
            action_name = state_action.argmax()
        return action_name

    def check_fox_q_table_exist(self, action, reward):
        state_space = self.convert_state_to_numbers(self.state)
        if state_space not in self.fox_q_table[0]:
            new_row = {
                "state": str(state_space),
                "(0, 1, 0)": 0,
                "(1, 1, 0)": 0,
                "(1, 0, 0)": 0,
                "(1, -1, 0)": 0,
                "(0, -1, 0)": 0,
                "(-1, -1, 0)": 0,
                "(-1, 0, 0)": 0,
                "(-1, 1, 0)": 0,

                "(0, 1, 1)": 0,
                "(1, 1, 1)": 0,
                "(1, 0, 1)": 0,
                "(1, -1, 1)": 0,
                "(0, -1, 1)": 0,
                "(-1, -1, 1)": 0,
                "(-1, 0, 1)": 0,
                "(-1, 1, 1)": 0

            }
            action_column = str(action)  # 确保 action 被转化为字符串格式
            if action_column in new_row:
                new_row[action_column] = reward  # 动态设置对应列的 Q 值为 reward

            self.fox_q_table.loc[len(self.fox_q_table)] = new_row

    def check_goose_q_table_exist(self, action, reward):
        state_space = self.convert_state_to_numbers(self.state)
        if state_space not in self.fox_q_table[0]:
            new_row = {
                "state": str(state_space),
                "(0, 1, 0)": 0,
                "(1, 1, 0)": 0,
                "(1, 0, 0)": 0,
                "(1, -1, 0)": 0,
                "(0, -1, 0)": 0,
                "(-1, -1, 0)": 0,
                "(-1, 0, 0)": 0,
                "(-1, 1, 0)": 0
            }
            action_column = str(action)  # 确保 action 被转化为字符串格式
            if action_column in new_row:
                new_row[action_column] = reward  # 动态设置对应列的 Q 值为 reward

            self.fox_q_table.loc[len(self.fox_q_table)] = new_row


    def get_fox_reward(self):

        sheep_count_last = sum(row.count('S') for row in self.state_last)
        sheep_count_current = sum(row.count('S') for row in self.state)

        if sheep_count_current < sheep_count_last:
            return 10  # 羊减少一只，奖励 10

        # 获取狼的当前位置和羊的位置
        fox_positions = [(x, y) for x, row in enumerate(self.state) for y, cell in enumerate(row) if cell == 'W']
        sheep_positions = [(x, y) for x, row in enumerate(self.state) for y, cell in enumerate(row) if cell == 'S']

        # 检测狼是否靠近羊
        for fx, fy in fox_positions:
            min_distance_last = float('inf')
            min_distance_current = float('inf')

            for sx, sy in sheep_positions:
                # 计算曼哈顿距离
                distance_last = abs(fx - sx) + abs(fy - sy)
                distance_current = abs(fx - sx) + abs(fy - sy)

                # 更新到最近羊的距离
                min_distance_last = min(min_distance_last, distance_last)
                min_distance_current = min(min_distance_current, distance_current)

            # 如果最近的距离变小，奖励 2
            if min_distance_current < min_distance_last:
                return 2

        for fx, fy in fox_positions:
            # 检查上下左右是否是墙壁
            if (fx > 0 and self.state[fx - 1][fy] == " ") or \
                    (fx < len(self.state) - 1 and self.state[fx + 1][fy] == " ") or \
                    (fy > 0 and self.state[fx][fy - 1] == " ") or \
                    (fy < len(self.state[0]) - 1 and self.state[fx][fy + 1] == " "):
                return -1  # 紧挨墙壁，扣除 1

        return 0
        pass

    def get_goose_reward(self):
        # 获取狐狸的当前位置和鹅的位置
        fox_positions = [(x, y) for x, row in enumerate(self.state) for y, cell in enumerate(row) if cell == 'F']
        goose_positions = [(x, y) for x, row in enumerate(self.state) for y, cell in enumerate(row) if cell == 'G']

        for gx, gy in goose_positions:
            min_distance = float('inf')
            closest_fox = None

            for fx, fy in fox_positions:
                distance = abs(fx - gx) + abs(fy - gy)
                if distance < min_distance:
                    min_distance = distance
                    closest_fox = (fx, fy)

            if min_distance < 2:  # 如果距离狐狸小于2单位
                # 查找鹅的目标位置
                directions = [
                    (gx, gy + 1),  # 右
                    (gx, gy - 1),  # 左
                    (gx - 1, gy),  # 上
                    (gx + 1, gy)  # 下
                ]

            for target_x, target_y in directions:
                # 确保目标位置在棋盘内
                if 0 <= target_x < len(self.state) and 0 <= target_y < len(self.state[0]):
                    # 如果目标位置是空地且到达目标后，左右或上下两个方向为空格
                    if self.state[target_x][target_y] == ".":
                        empty_count = 0
                        # 检查左右或上下两个方向
                        if (0 <= target_x < len(self.state) and target_y - 1 >= 0 and self.state[target_x][
                            target_y - 1] == "."):
                            empty_count += 1
                        if (0 <= target_x < len(self.state) and target_y + 1 < len(self.state[0]) and
                                self.state[target_x][target_y + 1] == "."):
                            empty_count += 1
                        if 0 <= target_x + 1 < len(self.state) and self.state[target_x + 1][target_y] == ".":
                            empty_count += 1
                        if empty_count == 2:
                            return -5  # 靠近狐狸并且到达一条直线的两个方向为空格，惩罚 -5

                # 向鹅群靠拢的奖励
                return 1

                # 前往的位置靠近狐狸，但三个方向挨着鹅或为空格
            for fx, fy in fox_positions:
                directions = [
                    (fx + 1, fy),  # 下
                    (fx - 1, fy),  # 上
                    (fx, fy + 1),  # 右
                    (fx, fy - 1)  # 左
                ]
                # 检查三个方向是否挨着鹅或为空格
                for target_x, target_y in directions:
                    if 0 <= target_x < len(self.state) and 0 <= target_y < len(self.state[0]):
                        # 目标位置是否是空地或接近鹅
                        if self.state[target_x][target_y] == "." or (
                                0 <= target_x < len(self.state) and 0 <= target_y < len(self.state[0]) and
                                self.state[target_x][target_y] == "G"):
                            return 5  # 靠近狐狸并且三个方向挨着鹅或为空格，奖励 +5

            # 距离大于2单位时向狐狸靠拢 +1
        if min_distance > 2:
            return 1

        return 0  # 默认奖励为 0

    def start(self):
        self.build_Q_table()

        for i in range(500):
            self.round += 1
            self.state_last = self.state
            if self.round % 2 == 1:
                action = self.play_fox()

                reward = self.get_fox_reward()

                self.check_fox_q_table_exist(action, reward)

            else:
                action = self.play_goose()
                reward = self.get_goose_reward()

                self.check_goose_q_table_exist(action, reward)

    def fox_try_kill(self):

        close_list = []
        plan_list = {}

        for dx, dy in self.fox_action_list:
            adjacent_fox = (self.fox.fox_x + dx, self.fox.fox_y + dy)
            for goose in self.goose_list:
                if adjacent_fox[0] == goose.goose_x and adjacent_fox[1] == goose.goose_y:
                    opposite_goose = (goose.goose_x - dx, goose.goose_y - dy)

                    if self.state[opposite_goose[0]][opposite_goose[1]] == ".":
                        close_list.append(opposite_goose)
        if len(close_list) > 1:
            for location in close_list:
                future_step = self.predict_future(location[0], location[1])
                plan_list[location] = future_step

        sorted_plan = sorted(plan_list.items(), key=lambda item: item[1], reverse=True)
        return list(sorted_plan.items())[0]

    def predict_future(self, current_x, current_y, future_kill=0):
        """
        预测狐狸从当前位置开始，能连续吃掉多少个鹅 这里有可能有bug
        """
        # 初始最大步数
        max_kill = future_kill

        # 遍历所有方向
        for dx, dy in self.fox_action_list:
            future_fox = (current_x + dx, current_y + dy)
            for goose in self.goose_list:
                if future_fox[0] == goose.goose_x and future_fox[1] == goose.goose_y:
                    # 如果发现可以吃掉鹅，检查反方向是否为空
                    opposite_goose = (goose.goose_x - dx, goose.goose_y - dy)
                    if self.state[opposite_goose[0]][opposite_goose[1]] == ".":
                        # 可以吃掉鹅，递归预测
                        max_kill = max(max_kill,
                                       self.predict_future(opposite_goose[0], opposite_goose[1], future_kill + 1))

        return max_kill

    # 执行一次移动
    def make_move(self, animal, action):
        if animal is Fox:
            current_x, current_y = animal.fox_x, animal.fox_y
            if action[2] == 1:  # kill
                self.state[current_x][current_y] = "."
                self.state[current_x + action[0]][current_y + action[1]] = "."
                self.state[current_x + action[0] + action[0]][current_y + action[1] + action[1]] = "F"
                self.fox.fox_x = current_x + action[0] + action[0]
                self.fox.fox_y = current_x + action[1] + action[1]
                for goose in self.goose_list:
                    if (goose.goose_x == (current_x + action[0])) and (goose.goose_y == (current_y + action[1])):
                        self.goose_list.remove(goose)
            else:
                self.state[current_x][current_y] = "."
                self.state[current_x + action[0]][current_y + action[1]] = "F"
                self.fox.fox_x = [current_x + action[0]]
                self.fox.fox_y = [current_y + action[1]]





        if animal is Goose:
            current_x, current_y = animal.goose_x, animal.goose_y
            self.state[current_x][current_y] = "."
            self.state[current_x + action[0]][current_y + action[1]] = "G"
            for goose in self.goose_list:
                if goose.goose_x == animal.goose_x and goose.goose_y == animal.goose_y:
                    goose.goose_x = current_x + action[0]
                    goose.goose_y = current_y + action[1]

    # 自动操作狐狸
    def play_fox(self):
        # if self.is_auto:
            action = self.fox_select_action(self.state, self.fox_q_table)
            self.make_move(self.fox, action)
            return action

    # 自动操作鹅
    def play_goose(self):
        # if self.is_auto:
            action = self.goose_select_action(self.state, self.goose_q_table)
            self.make_move(self.goose_list, action)
            return action


Player().start()
