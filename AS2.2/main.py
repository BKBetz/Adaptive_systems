from Maze import Maze
from Agent import Agent
from Policy import Policy

maze = Maze()
maze.create_maze()
policy = Policy(maze)

agent = Agent(maze, policy, (3, 2))
agent.value_iteration()
# agent.temporal_difference(0.5, 1, 10)
# agent.temporal_difference(0.5, 0.5, 10)
# agent.sarsa(0.25, 1, 0.1, 200000)
# agent.sarsa(0.25, 0.9, 0.1, 20000)
agent.q_learning(0.25, 1, 0.1, 50000)
# agent.q_learning(0.5, 0.9, 0.1, 50000)

