import numpy as np
import matplotlib.pyplot as plt


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
        step = self.policy.decide_action_utility(self.current_pos, next_states)
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
        """
        Evaluate the policy using temporal difference learning
        :param
            lr: learning rate
            discount: the discount
            epochs: amount of episodes
        """
        # the optimal policy episode
        episode = [0, 2, 0, 0, 3, 3]
        for i in range(epochs):
            for step in episode:
                next_state = self.maze.step(self.current_pos, step)
                # if state is terminal, episode ends
                if self.maze.states[next_state][2] is False:
                    c_value = self.maze.states[self.current_pos][1]
                    reward, next_value = self.maze.states[next_state][0], self.maze.states[next_state][1]
                    new_value = c_value + lr * (reward + (discount * next_value) - c_value)
                    self.maze.states[self.current_pos][1] = new_value
                    self.current_pos = next_state
                else:
                    print("terminal state", next_state)
                    # reset agent position back to start position for next episode
                    self.set_current_pos((3, 2))

            self.show_values(i + 1)

    def sarsa(self, lr, discount, epsilon, epochs):
        """
        create optimal policy using sarsa on policy control
        :param
            lr: learning rate
            discount: discount
            epsilon: randomness using epsilon
            epochs: amount of episodes
        :return:
        """
        for i in range(epochs):
            c_state = self.maze.states[self.current_pos]
            action = self.policy.decide_action_value(c_state, epsilon)
            while c_state[2] is False:
                next_pos = self.maze.step(self.current_pos, action)
                next_state = self.maze.states[next_pos]
                next_action = self.policy.decide_action_value(next_state, epsilon)

                c_state[3][action] = c_state[3][action] + lr * (next_state[0] + discount * next_state[3][next_action] - c_state[3][action])
                action = next_action
                self.current_pos = next_pos
                c_state = next_state

            self.set_current_pos((3, 2))

        self.show_sarsa_policy(epochs, lr, epsilon, discount)

    def q_learning(self, lr, discount, epsilon, epochs):
        """
        create optimal policy using Q learning off policy control
        :param
            lr: learning rate
            discount: discount
            epsilon: randomness using epsilon
            epochs: amount of episodes
        :return:
        """
        for i in range(epochs):
            c_state = self.maze.states[self.current_pos]
            while c_state[2] is False:
                action = self.policy.decide_action_value(c_state, epsilon)
                next_pos = self.maze.step(self.current_pos, action)
                next_state = self.maze.states[next_pos]
                best_action_value = max(next_state[3])
                c_state[3][action] = c_state[3][action] + lr * (next_state[0] + discount * best_action_value - c_state[3][action])
                self.current_pos = next_pos
                c_state = next_state

            self.set_current_pos((3, 2))

        self.show_ql_policy(epochs, lr, epsilon, discount)

    def show_sarsa_policy(self, epochs, lr, epsilon, discount):
        maze = []
        matrix = []
        actions = {0: "up   ", 1: "down ", 2: "left ", 3: "right"}
        for state in self.maze.states:
            if self.maze.states[state][2] is False:
                max_action = max(self.maze.states[state][3])
                max_index = self.maze.states[state][3].index(max_action)
                maze.append(actions[max_index])
                matrix.append(self.maze.states[state][3])
            else:
                maze.append("Terminal")
                matrix.append([0, 0, 0, 0])

        test = np.array(matrix)
        test = test.reshape(4, 4, 4)

        fig, axs = plt.subplots(1, test.shape[0], figsize=(12, 4))

        for i in range(test.shape[0]):
            axs[i].imshow(test[i], cmap='viridis')
            axs[i].set_title(f'row {i}')
            for j in range(test.shape[1]):
                for k in range(test.shape[2]):
                    axs[i].text(k, j, str(round(test[i, j, k], 2)), ha='center', va='center', color='w')
        plt.show()

        maze = np.array(maze)
        text = """Policy after {ep} episodes
learning rate {lr}
discount {dc}
epsilon {es}"""
        # print(text.format(lr=lr, ep=epochs, es=epsilon, dc=discount))
        # print(maze.reshape(4, 4))

    def show_ql_policy(self, epochs, lr, epsilon, discount):
        maze = []
        matrix = []
        actions = {0: "up   ", 1: "down ", 2: "left ", 3: "right"}

        for state in self.maze.states:
            if self.maze.states[state][2] is False:
                max_action = max(self.maze.states[state][3])
                max_index = self.maze.states[state][3].index(max_action)
                maze.append(actions[max_index])
                matrix.append(self.maze.states[state][3])
            else:
                maze.append("Terminal")
                matrix.append([0, 0, 0, 0])

        test = np.array(matrix)
        test = test.reshape(4, 4, 4)

        fig, axs = plt.subplots(1, test.shape[0], figsize=(12, 4))

        for i in range(test.shape[0]):
            axs[i].imshow(test[i], cmap='viridis')
            axs[i].set_title(f'row {i}')
            for j in range(test.shape[1]):
                for k in range(test.shape[2]):
                    axs[i].text(k, j, str(round(test[i, j, k], 2)), ha='center', va='center', color='w')
        plt.show()

        maze = np.array(maze)
        text = """Policy after {ep} episodes
        learning rate {lr}
        discount {dc}
        epsilon {es}"""
        print(text.format(lr=lr, ep=epochs, es=epsilon, dc=discount))
        print(maze.reshape(4, 4))

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
                best_move = self.policy.decide_action_utility(state, next_states)
                b_moves[state] = best_move
            maze[state] = self.maze.states[state][1]

        print(maze)
        # print(b_moves)


