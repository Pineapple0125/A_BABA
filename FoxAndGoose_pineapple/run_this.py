from brain import DeepQNetwork
from FG_env import Env


def run_battle():
    step = 0
    for episode in range(20):
        # initial observation
        observation = env.reset()
        while True:
            # fresh env
            env.reset()

            # RL choose action based on observation
            # 狐狸先行动
            action_fox = RL.choose_action(observation, "fox")
            print(action_goose)

            # RL take action and get next observation and reward (狐狸行动)
            observation_fox, reward_fox, done_fox = env.fox_step(action_fox)
            RL.store_transition(observation, action_fox, reward_fox, observation_fox)  # 存储狐狸的记忆

            # 如果步数大于200并且是5的倍数，学习
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # 更新为狐狸执行后的状态
            observation = observation_fox

            # 判断是否结束了
            if done_fox:
                break

            # 鹿（鹅）行动
            action_goose = RL.choose_action(observation, "goose")
            print(action_goose)

            # RL take action and get next observation and reward (鹅行动)
            observation_goose, reward_goose, done_goose = env.goose_step(action_goose)
            RL.store_transition(observation, action_goose, reward_goose, observation_goose)  # 存储鹅的记忆

            # 如果步数大于200并且是5的倍数，学习
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # 更新为鹅执行后的状态
            observation = observation_goose

            # 判断是否结束了
            if done_goose:
                break

            step += 1

    # end of game
    print('Game Over')


if __name__ == "__main__":
    # maze game
    env = Env()
    RL = DeepQNetwork(env.actions_fox, env.actions_goose, env.features, len(env.goose_list),
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    run_battle()
    RL.plot_cost()
