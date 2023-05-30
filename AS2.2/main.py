from Maze import Maze
from Agent import Agent
from Policy import Policy

maze = Maze()
maze.create_maze()
policy = Policy()

agent = Agent(maze, policy, (3, 2))
# agent.value_iteration()
# agent.temporal_difference(1, 0.5, 10)
agent.sarsa(0.5, 0.5, 0.1, 1)

