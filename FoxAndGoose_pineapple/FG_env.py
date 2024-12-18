import time

import numpy as np


class Fox:
    def __init__(self, fox_x, fox_y):
        self.fox_x = fox_x
        self.fox_y = fox_y


class Goose:
    def __init__(self, goose_x, goose_y, goose_index):
        self.goose_x = goose_x
        self.goose_y = goose_y
        self.index = goose_index


class Env(object):
    def __init__(self):

        self.MAZE_H = 7  # 7x7网格的高度
        self.MAZE_W = 7  # 7x7网格的宽度
        self.UNIT = 1  # 网格单位大小
        self.fox_pos = np.array([3, 3])

        self.goose_positions = np.array([
            [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2],
            [3, 0], [3, 1], [3, 2], [4, 0], [4, 1], [4, 2]
        ])  # 鹅的位置列表，共15只

        self.convert_num = {
            ' ': 0,  # 空白
            '.': 1,  # 可能表示障碍或地面
            'F': 2,  # 狐狸
            'G': 3  # 鹅
        }
        self.state = [
            [' ', ' ', '.', '.', '.', ' ', ' '],  # 0  1  2  3  4  5  6
            [' ', ' ', '.', '.', '.', ' ', ' '],  # 7  8  9 10 11 12 13
            ['.', '.', '.', '.', '.', '.', '.'],  # 14 15 16 17 18 19 20
            ['G', '.', '.', 'F', '.', '.', 'G'],  # 21 22 23 24 25 26 27
            ['G', 'G', 'G', 'G', 'G', 'G', 'G'],  # 28 29 30 31 32 33 34
            [' ', ' ', 'G', 'G', 'G', ' ', ' '],  # 35 36 37 38 39 40 41
            [' ', ' ', 'G', 'G', 'G', ' ', ' ']  # 42 43 44 45 46 47 48
        ]

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


        self.actions_fox = len(self.fox_action_list)
        self.actions_goose = len(self.goose_action_list)

        self.fox = Fox(3, 3)



        self.goose_list = [Goose(goose_x=x, goose_y=y, goose_index=index)
                           for index, (x, y) in enumerate(
                [(3, 0), (3, 6), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
                 (5, 2), (5, 3), (5, 4), (6, 2), (6, 3), (6, 4)]
            )]

        self.state_last = []
        self.features = (7, 7)

    def reset(self):
        # 初始化地图的状态
        self.state = [
            [' ', ' ', '.', '.', '.', ' ', ' '],  # 0  1  2  3  4  5  6
            [' ', ' ', '.', '.', '.', ' ', ' '],  # 7  8  9 10 11 12 13
            ['.', '.', '.', '.', '.', '.', '.'],  # 14 15 16 17 18 19 20
            ['G', '.', '.', 'F', '.', '.', 'G'],  # 21 22 23 24 25 26 27
            ['G', 'G', 'G', 'G', 'G', 'G', 'G'],  # 28 29 30 31 32 33 34
            [' ', ' ', 'G', 'G', 'G', ' ', ' '],  # 35 36 37 38 39 40 41
            [' ', ' ', 'G', 'G', 'G', ' ', ' ']  # 42 43 44 45 46 47 48
        ]

        # 重置狐狸的位置
        self.fox = Fox(3, 3)
        self.fox_pos = np.array([3, 3])
        self.state[3][3] = 'F'  # 在地图上放置狐狸

        # 重置鹅的位置（15只鹅）
        self.goose_list = [Goose(goose_x=x, goose_y=y, goose_index=index)
                           for index, (x, y) in enumerate(
                [(3, 0), (3, 6), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
                 (5, 2), (5, 3), (5, 4), (6, 2), (6, 3), (6, 4)]
            )]

        # 在地图上放置鹅
        for goose in self.goose_list:
            self.state[goose.goose_x][goose.goose_y] = 'G'

        print("SET")
        print(np.array([[self.convert_num[cell] for cell in row] for row in self.state]))
        # 返回当前的状态
        return np.array([[self.convert_num[cell] for cell in row] for row in self.state])


    def check_game_done(self):
        fox_position = None
        goose_count = 0

        # 遍历地图，记录鹅的数量和狐狸位置
        for i in range(len(self.state)):
            for j in range(len(self.state[0])):
                if self.state[i][j] == 'F':
                    fox_position = (i, j)
                elif self.state[i][j] == 'G':
                    goose_count += 1

        # 检查鹅的数量
        if goose_count < 4:
            print("Fox wins! Goose count is less than 4.")
            return True  # 游戏结束

        # 如果没有找到狐狸，异常处理
        if not fox_position:
            raise ValueError("Fox not found on the map!")

        # 获取狐狸的位置
        x, y = fox_position

        # 定义四个方向：上、下、左、右
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        fox_surrounded = True  # 假设狐狸被包围

        for dx, dy in directions:
            # 检查狐狸四周的格子
            nx, ny = x + dx, y + dy

            if 0 <= nx < len(self.state) and 0 <= ny < len(self.state[0]):
                if self.state[nx][ny] != 'G' and self.state[nx][ny] != ' ':
                    # 如果有一个方向不被 G 或 空格 占据，狐狸未被包围
                    fox_surrounded = False
                    break

                # 检查反方向
                bx, by = x - dx, y - dy
                if 0 <= bx < len(self.state) and 0 <= by < len(self.state[0]):
                    if self.state[bx][by] != 'G' and self.state[bx][by] != ' ':
                        fox_surrounded = False
                        break

        # 如果狐狸被完全包围
        if fox_surrounded:
            print("Goose wins! Fox is trapped.")
            return True  # 游戏结束

        return False  # 游戏继续







    def fox_step(self, action):
        current_x, current_y = self.fox.fox_x, self.fox.fox_y
        self.state_last = self.state
        step = self.fox_action_list[action]
        print()
        print(step)
        print()
        if step[2] == 1:  # kill
            if (self.state[current_x + step[0]][current_y + step[1]] == "G") and (self.state[current_x + step[0] + step[0]][current_y + step[1] + step[1]] == "."):
                self.state[current_x][current_y] = "."
                self.state[current_x + step[0]][current_y + step[1]] = "."
                self.state[current_x + step[0] + step[0]][current_y + step[1] + step[1]] = "F"
                self.fox.fox_x = current_x + step[0] + step[0]
                self.fox.fox_y = current_x + step[1] + step[1]
                for goose in self.goose_list:
                    if (goose.goose_x == (current_x + step[0])) and (goose.goose_y == (current_y + step[1])):
                        self.goose_list.remove(goose)
        else:
            if self.state[current_x + step[0]][current_y + step[1]] == ".":
                self.state[current_x][current_y] = "."
                self.state[current_x + step[0]][current_y + step[1]] = "F"
                self.fox.fox_x = [current_x + step[0]]
                self.fox.fox_y = [current_y + step[1]]

        reward = self.get_fox_reward()

        done = self.check_game_done()



        return np.array([[self.convert_num[cell] for cell in row] for row in self.state]), reward, done





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



    def goose_step(self, action):
        print("GGGG")
        print(action)
        print("GGGGG")

        for goose in self.goose_list:
            if goose.index == action.index:
                current_x, current_y = goose.goose_x, goose.goose_y
                self.state[current_x][current_y] = "."
                self.state[current_x + action[0]][current_y + action[1]] = "G"
                if goose.goose_x == goose.goose_x and goose.goose_y == goose.goose_y:
                    goose.goose_x = current_x + action[0]
                    goose.goose_y = current_y + action[1]

        reward = self.get_goose_reward()

        done = self.check_game_done()

        return np.array([[self.convert_num[cell] for cell in row] for row in self.state]), reward, done




    def get_goose_reward(self):
        #  获取狐狸的当前位置和鹅的位置
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






