"""Maze the agents walks in"""

import numpy as np

class Maze:
    """The Maze, contains functions for finding bordering and updating states, making a step and creating a new maze"""
    def __init__(self):
        self.states = ""
        self.actions = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}

    def get_states(self, state):
        """
            Return the bordering states of a state
        :return:
            state: a given state
        """
        states = {}
        for action in self.actions.keys():
            # for each action, fill the states dict with the coordinate as key and list as value
            states[self.step(state, action)] = self.states[self.step(state, action)]

        return states

    def create_maze(self):
        """
            Create a maze using the maze_format given at the start. Turns each value of that maze into a state with
            coordinates and rewards
        """
        matrix = np.zeros((4, 4))
        rewards = np.array([[-1, -1, -1, 40],
                  [-1, -1, -10, -10],
                  [-1, -1, -1, -1],
                  [10, -2, -1, -1]])
        self.states = self.create_states(matrix, rewards)

    def create_states(self, maze: np.array, rewards: np.array):
        """
        Gives each cell in a maze: coordinates, a reward, a value and a boolean for terminal state.
        At the end, two states get picked to be terminal states
        :param
            maze: The empty maze
            rewards: the rewards for each state
        :return:
            states: The new maze with states
        """
        states = {}
        # easy way to get all coordinates using numpy
        for x, y in np.ndindex(maze.shape):
            coord = (x, y)
            states[coord] = [rewards[x][y], 0, False, [0, 0, 0, 0]]

        # set terminal states
        states[(0, 3)][2] = True
        states[(3, 0)][2] = True

        return states

    def step(self, c_state: tuple, action: int):
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
        if new_pos in self.states.keys():
            return new_pos
        else:
            return c_state

    def update_states(self, new_values):
        """
            Updates old values with new values after a iteration

            :param
                new_values: new values after a value iteration
        """

        for coords, value in new_values.items():
            self.states[coords][1] = value
