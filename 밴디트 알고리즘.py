import numpy as np
import matplotlib.pyplot as plt
class Bandit:
    def __init__(self, arms=10): # arms = 슬롯머신 대수
        self.rates = np.random.rand(arms) # 슬롯머신 각각의 승률 설정(무작위)

    def play(self, arm): #arm: 몇 번째 슬롯머신을 플레이할지
        rate = self.rates[arm] #arm번째 머신의 승률
        if rate > np.random.rand():
            return 1
        else:
            return 0


#에이전트 구현
class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon # 탐색할 확률
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):#슬롯머신의 가치 추정
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self): # 행동 선택 (엡실론 - 탐욕 정책)
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs)) #무작위 행동 선택
        return np.argmax(self.Qs) #탐욕 행동 선택


runs = 200
steps = 1000
epsilon = 0.1
all_rates = np.zeros((runs, steps))




for run in range(runs):
    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    rates = []  # 승률

    for step in range(steps):
        action = agent.get_action() #1. 행동 선택
        reward = bandit.play(action) #2. 실제로 플레이하고 보상 획득
        agent.update(action, reward) #3. 행동과 보상을 통해 학습
        total_reward += reward

        total_reward += reward
        rates.append(total_reward / (step + 1)) # 현재까지의 승률 저장

    all_rates[run] = rates

avg_rates = np.average(all_rates, axis=0)


#단계별 승률
plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(avg_rates)
plt.show()