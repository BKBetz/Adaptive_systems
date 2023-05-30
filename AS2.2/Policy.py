import random as rd


class Policy:
    """
        The policy of the agent. contains the value function and a action function
    """
    def __init__(self):
        """
        -1 means that there is no better action. Meaning the either:
        1: No states are better than the state that's being used
        2: Reached a terminal state
        """
        self.actions = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1], -1: [0, 0]}

    def value_func(self, next_states, discount):
        """
        Calculate the value given the states bordering current state

        :param
            next_states: The states bordering the current states
            discount: The discountfactor
        :return:
        """

        values = []
        for coord, items in next_states.items():
            sum = items[0] + discount * items[1]
            values.append(sum)

        new_value = max(values)

        return new_value

    def select_random_action(self):
        """
        Select a random action
        :return:
            a random choice for an action
        """
        return rd.choice([0, 1, 2, 3])

    def decide_action(self, state, next_states):
        """
        A function for finding the best action using the current position and
        finding the bordering state with the highest utility.
        :param
            state: current state
            next_states: bordering states
        :return:
            an action
        """
        # find the state with the highest utility

        state_utilities = {}

        for n_state in next_states:
            state_utilities[n_state] = next_states[n_state][0] + next_states[n_state][1]

        best_state = max(state_utilities, key=state_utilities.get)

        move = []
        for i in range(0, len(state)):
            # the move that needs to be made to go from the current state tot the best state
            move.append(best_state[i] - state[i])

        # the key equal to the move that needs to be made
        action = [k for k, v in self.actions.items() if v == move]
        return action[0]







