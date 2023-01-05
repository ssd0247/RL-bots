# import the requierd libraries
import numpy as np

# define the shape of the environment (ie, the states)
env_rows, env_cols = 11, 11
# define actions, numeric action codes: 0=up, 1=right, 2=down, 3=left
actions = ['up','right','down','left']

# Create a 3D array to hold the current Q-values for each possible
# state, action pair : (S, A).
# The state is given by the row_index, and the col_index....
# And the action could be one of the four directions in which the robot
# can navigate.
q_values = np.zeros((env_rows, env_cols, len(actions)))

# Create a 2D array to gold the rewards for each state.
rewards = np.full((env_rows, env_cols), -100.0)
rewards[0,5] = 100.0 # the packaging area(ie, the goal)
# define the aisle locations, where the robot can move to.
aisles = {}
aisles[1] = [i for i in range(1, 10)]
aisles[2] = [1, 7, 9]
aisles[3] = [i for i in range(1, 8)]
aisles[3].append(9)
aisles[4] = [3, 7]
aisles[5] = [i for i in range(11)]
aisles[6] = [5]
aisles[7] = [i for i in range(1, 10)]
aisles[8] = [3, 7]
aisles[9] = [i for i in range(11)]
# set the rewards for the aisle locations.
for row_index in range(1, 10):
    for col_index in aisles[row_index]:
        rewards[row_index, col_index] = -1.0
# print rewards matrix
for row in rewards:
    print(row)


# TRAIN THE MODEL
#
# 1. Choose a random, non-terminal state for the agent to begin his new episode.
#
# 2. Choose an action for the current state. Actions are choosen using `EPSILON-GREEDY`
#   algorithm. This algorithm usually chooses the most promising action, but occasionally
#   will choose a lesser promising one, with the hopes of exploring the environment, to somehow
#   land in a spot having good values/rewards associated with it.
#
# 3. Perform the action, and transition to the next state.
# 
# 4. Receive the reward for moving to the new state, and calculate the temporal difference.
# 
# 5. Update the Q-value for the previous state and action-pair.
# 
# 6. If the new(current) state is a terminal state, goto #1, else, goto #2.
# 
# 7. The entire process is repeated across 1000 episodes.

def is_terminal_state(row_index, col_index):
    """Check whether a given position, in the environment,
    corresponds to a terminal or a non-terminal state."""
    if rewards[row_index, col_index] == -1.0:
        return False
    return True

def get_starting_location():
    """Will choose a random, non-terminal starting location."""
    curr_row_index = np.random.randint(env_rows)
    curr_col_index = np.random.randint(env_cols)
    while is_terminal_state(curr_row_index, curr_col_index):
        curr_row_index = np.random.randint(env_rows)
        curr_col_index = np.random.randint(env_cols)
    return curr_row_index, curr_col_index

def get_next_action(curr_row_index, curr_col_index, epsilon):
    """Epsilon-Greedy Algorithm, that controls the exploration vs 
    exploitation ratio, based on epsilon's fractional value."""
    if np.random.random() < epsilon:
        return np.argmax(q_values[curr_row_index, curr_col_index])
    else: # choose a random action
        return np.random.randint(len(actions))

def get_next_location(curr_row_index, curr_col_index, action_index):
    """Get the next location of the agent, based on the current state
    and the corresponding action executed."""
    new_row_index = curr_row_index
    new_col_index = curr_col_index
    if actions[action_index] == 'up' and curr_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and curr_col_index < env_cols - 1:
        new_col_index += 1
    elif actions[action_index] == 'down' and curr_row_index < env_rows - 1:
        new_row_index += 1
    elif actions[action_index] == 'left' and curr_col_index > 0:
        new_col_index -= 1
    return new_row_index, new_col_index

def train(episodes=1000, epsilon=0.9, discount_factor=0.9, learning_rate=0.9):
    """Configures the hyperparameters, and other settings to run the
    Q-learning algorithm."""
    global q_values
    # define the training parameters
    epsilon = epsilon 
    discount_factor = discount_factor # for future rewards
    learning_rate = learning_rate # the rate at which agent learns
    # run through large numbers of training episodes.
    print("Training Starts....")
    for _ in range(episodes):
        # get the starting location for this episode
        row_index, col_index = get_starting_location()
        # continue taking actions, until we reach the destination / terminal state
        while not is_terminal_state(row_index, col_index):
            action_index = get_next_action(row_index, col_index, epsilon)
            old_row_index, old_col_index = row_index, col_index
            row_index, col_index = get_next_location(old_row_index, old_col_index, action_index)
            # receive reward of moving to the new state
            reward = rewards[row_index, col_index]
            old_q_value = q_values[old_row_index, old_col_index, action_index]
            temporal_diff = reward + (discount_factor * np.max(q_values[row_index, col_index]))
            new_q_value = old_q_value + (learning_rate * temporal_diff)
            q_values[old_row_index, old_col_index, action_index] = new_q_value
    print("Training Complete !!!")

def get_shortest_path(start_row_index, start_col_index):
    """Shortest path calculation, between any location in the warehouse
    where the robot is allowed to travel, and the end-goal location, ie,
    the packaging area."""
    if is_terminal_state(start_row_index, start_col_index):
        return []
    else:
        curr_row_index, curr_col_index = start_row_index, start_col_index
        shortest_path = []
        shortest_path.append([curr_row_index, curr_col_index])
        # continue moving along the path until we reach the goal
        while not is_terminal_state(curr_row_index, curr_col_index):
            # get the best action to take
            action_index = get_next_action(curr_row_index, curr_col_index, 1.0)
            # move to the next location on the path, and add the new location to the path list
            curr_row_index, curr_col_index = get_next_location(curr_row_index, curr_col_index, action_index)
            shortest_path.append([curr_row_index, curr_col_index])
        return shortest_path

if __name__ == '__main__':
    train()
    #print(get_shortest_path(3, 9))
    #print(get_shortest_path(5, 0))
    #print(get_shortest_path(9, 5))

    ## when the robot needs to pick another item, thus essentially traversing
    ## the path in reverse.
    #path = get_shortest_path(5, 2)
    #path.reverse()
    #print(path)