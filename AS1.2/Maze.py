"""Maze the agents walks in"""

import numpy as np
from typing import List
import random as rd


class Maze:
    """The Maze"""
    def __init__(self, maze_format: List, reward: int):
        self.maze = maze_format
        self.reward = reward
        self.actions = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}

    def create_maze(self):
        """
            Create a maze using the maze_format given at the start. Turns each value of that maze into a state with
            coordinates and rewards
        """
        matrix = np.zeros((self.maze[0], self.maze[1]))
        self.maze = self.create_states(matrix)

    def create_states(self, maze: np.array):
        """
        Gives each value in a maze: coordinates, a reward and a boolean for terminal state.
        At the end, two states get randomly picked to be terminal states
        :param
            maze: The empty maze
        :return:
            states: The new maze with states
        """
        states = {}
        # easy way to get all coordinates using numpy
        for x, y in np.ndindex(maze.shape):
            coord = (x, y)
            states[coord] = [self.reward, False]

        # determine terminal states
        terminal_states = rd.choices(list(states), k=2)
        for state in terminal_states:
            states[state][1] = True

        return states

    def step(self, c_state: list, action: list):
        """
        Determine the new position given a state and a action
        :param
            c_state: the current state the agent is in
            action: The action that will be used (up, down, left, right)
        :return:
            new_pos: The new coordinates if it is possible (within the maze)
            c_state: The current state if the new position is not possible (not within the maze)
        """

        # new pos = sum from current state plus the step you make with a action
        new_pos = tuple(sum(x) for x in zip(c_state, self.actions[action]))
        if new_pos in self.maze.keys():
            return new_pos
        else:
            return c_state
