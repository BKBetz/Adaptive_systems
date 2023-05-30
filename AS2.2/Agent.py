import numpy as np


class Agent:
    """The Agent"""
    def __init__(self, maze, policy, s_pos):
        self.maze = maze
        self.policy = policy
        self.start_pos = s_pos
        self.current_pos = s_pos

    def act(self):
        """
            show the current position and positions after taking a action.
            It is used AFTER the value iteration has taken place.
        """
        print("current position", self.current_pos)
        next_states = self.maze.get_states(self.current_pos)
        step = self.policy.decide_action(self.current_pos, next_states)
        if step == -1:
            print("Terminal stage reached")

        else:
            new_pos = self.maze.step(self.current_pos, step)
            self.current_pos = new_pos
            print("new position", self.current_pos)

    def set_current_pos(self, pos):
        self.current_pos = pos

    def value_iteration(self):
        """
            Iterate through each state and calculate the utility of each state
        """
        # force in the while loop
        delta = 1
        iteration = 0
        while delta > 0.01:
            delta = 0
            new_values = {}
            # loop through all states
            for state, values in self.maze.states.items():
                # check if state is not terminal state
                if not values[2]:
                    # get bordering states
                    next_states = self.maze.get_states(state)
                    # calculate value
                    new_value = self.policy.value_func(next_states, 1)
                    new_values[state] = new_value
                    # update delta
                    delta = max(delta, abs(values[1] - new_value))
            # after all states have been updates, update the entire maze at once.
            self.maze.update_states(new_values)
            iteration += 1
            """
                this function is commented out to prevent showing to much prints in the terminal.
                uncomment to see the utility of all states after each iteration 
                and also see which action is best in each state.
            """
            # self.show_values(iteration)

    def temporal_difference(self, lr, discount, epochs):
        episode = [0, 2, 0, 0, 3, 3]
        for i in range(epochs):
            for step in episode:
                next_state = self.maze.step(self.current_pos, step)
                if self.maze.states[next_state][2] is False:
                    c_value = self.maze.states[self.current_pos][1]
                    reward, next_value = self.maze.states[next_state][0], self.maze.states[next_state][1]
                    new_value = c_value + lr * (reward + (discount * next_value) - c_value)
                    self.maze.states[self.current_pos][1] = new_value
                    self.current_pos = next_state
                else:
                    print("terminal state", next_state)
                    self.set_current_pos((3, 2))

            self.show_values(i + 1)

    def sarsa(self):
        pass

    """
        SARSA IS POLICY OPBOUWEN, ALLES Q VALUES BEGINNNE OP 0, VOLG BEREKENING OP CANVAS. VOLG POLICY EPSILON GREEDY
        SARSA PAKT 2 keer een random keuze, qlearing 1, A wordt bij SARSA A'
    """

    def show_values(self, iteration):
        """
        Hardcoded print function to show the utility and best action of each state at each iteration
        :param
            iteration:
        :return:
        """
        print(iteration)
        maze = np.zeros((4, 4))
        b_moves = np.empty((4, 4))
        for state in self.maze.states:
            if not self.maze.states[state][2]:
                next_states = self.maze.get_states(state)
                best_move = self.policy.decide_action(state, next_states)
                b_moves[state] = best_move
            maze[state] = self.maze.states[state][1]

        print(maze)
        # print(b_moves)


