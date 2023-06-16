import gym
from Agent import Agent
from Policy import Policy
from Memory import Memory
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model


env = gym.make("LunarLander-v2",
               continuous= False,
               enable_wind= True,
               render_mode="human")


memory = Memory(10000)
policy = Policy(1)
input_dms = env.observation_space.shape
actions = env.action_space.n
agent = Agent(policy, memory, 0.99)
agent.policy.create_model(input_dms, actions, 128, 128, 0.001)
scores = []
avg_scores = []

observation, info = env.reset(seed=42)
episodes = 200

for x in range(episodes):
    observation, info = env.reset()
    score = 0
    steps = 0
    terminated = False
    while not terminated:
        action = agent.policy.select_action(observation)
        new_observation, reward, terminated, truncated, info = env.step(action)
        score += reward
        steps += 1
        transition = (observation, action, reward, new_observation, terminated)
        agent.memory.store(transition)
        observation = new_observation
        if x > 2:
            agent.train()

    scores.append(score)
    # bereken gemiddelde score van de laatste 10 episodes
    avg = np.mean(scores[-100:])
    avg_scores.append(avg)

    print("episode", x, "score %.2f" % score, "average score %.2f" % avg, "epsilon %.2f" % agent.policy.epsilon, "steps", steps)


env.close()

fig, ax = plt.subplots()
ax.plot(np.arange(episodes), np.array(avg_scores), label='average score')
ax.plot(np.arange(episodes), np.array(scores), 'd', label='score')
plt.xlabel("episode")
plt.ylabel("score")
ax.legend()
plt.show()

# agent.policy.model.save("AS3.1")
