from Maze import Maze
from Agent import Agent
from Policy import Policy

maze = Maze()
maze.create_maze()
policy = Policy()

agent = Agent(maze, policy, (3, 2))
agent.value_iteration()
# after value iteration use act to show the agent moving to the best terminal state
# tip, comment the code below if u uncommented "show_utility" agent.py line 57
# agent.act()
# agent.act()
# agent.act()
# agent.act()
# agent.act()
# agent.act()
# agent.act()
