import math
import random
from multiprocessing import Process, Pool

# TODO: The experiments run, fast, but the results aren't getting accumulated inside shared
#       lists namely `results_greedy`,  `results_epsilon_greedy` ? WHY ?

## ----------- Reinforcement - Learning ------------ ##

# number_of_bandits = 2000, play them PARALLELY.
#
# number_of_steps = 1000, for each bandit.
#
# NOTE: Value estimations are done randomly, improve this ?
#
# action-values choosen according to two strategies :
#       A) Purely greedy -> best valued action choosen at each step, by every bandit there is (all 2000). Record their performances, average them out.
#       B) Epsilon greedy -> a small fraction of times (when ?) randomly choose some action ( remove the best action at that time, as if it is randomly chosen, then exploration mimics exploitation :O ) and generate the corresponding reward from the static distribution. Keeps track of the `action-estimates` of the actions taken.
#
# Compare the performances and plot the graph.
#
# At `t`th time-step, the reward that the choosen action a(t) generates is:
#   - value of the action q(a) + noise (sampled from gaussian with mean 0, variance 1).
#
# Action-values AREN'T UPDATED, BUT rather sampled from the gaussian distribution, 
# whenever it's time for EXPLORATION !, and then the noise is added to the one chosen.
#
# Stationary reward distribution, ie, mean and variance is fixe and all rewards are 
# *ASSUMED* to be from this distribution throughtout the time-steps and bandits.

def value_estimate_generation(n, bounds=(1, 100)):
    '''
    Generate the value estimates from static distribution.
    Standard deviation - 1.
    Mean - 0.
    '''
    lo, hi = bounds
    values = []
    for _ in range(n):
        val = random.randint(lo, hi)
        reward = 1 / (1 * math.sqrt(2 * math.pi)) * math.exp(-0.5*val**2) 
        values.append(reward)
    return values

def choose_the_best_action(values):
    '''
    Best value index == best action index.
    '''
    return values.index(max(values))

results_greedy = []
def run_greedy(num_actions, num_steps, n_prints=10):
    acc = [] # rewards accumulated at each step (for comparing performances)
    for step in range(num_steps):
        # Generate the action-value estimates
        values = value_estimate_generation(num_actions)
        # Choose the best action
        action_index = choose_the_best_action(values)
        # Reward generated
        reward = values[action_index]
        # accumulate the reward
        acc.append(reward)

        if step %  (num_steps // n_prints) == 0:
            print(f"Step : {step}\tReward : {reward}")
        
    results_greedy.append(sum(acc)) 

results_epsilon_greedy = []
def run_epsilon_greedy(num_actions, num_steps, n_prints=10, epsilon=0.01):
    acc = [] # rewards accumulated at each step (for comparing performances)
    explorations = 0 # count the number of explorations done !!

    # XXX: How to affirm, that a given step is the correct step to take the `exploration-path` ?
    #       Here I'm just checking at every 50th iteration that `epsilon` fraction are explorative.

    for step in range(1, num_steps+1, 1):
        # Generate the action-value estimates
        values = value_estimate_generation(num_actions)

        explored = False # toggle this if exploration is done

        # Check at each step, that explorations are done in sufficient number till now.
        if epsilon * step < explorations: # EXPLORATION
            explorations += 1
            explored = True
            len_values = len(values)
            random_value_index = random.randint(0, len_values-1)
            action_index = random_value_index
        else: # Choose the best action; EXPLOITATION
            action_index = choose_the_best_action(values) 
        
        # Reward generated
        reward = values[action_index]
        # accumulate the reward
        acc.append(reward)

        if step %  (num_steps // n_prints) == 0:
            print(f"Step : {step}\tReward : {reward}\tExploration : {explored}")
        
    results_epsilon_greedy.append(sum(acc))

def run_experiments(num_bandits, num_steps, num_actions):
   
    processes = []
    for _ in range(num_bandits):
        funcA = run_greedy
        funcB = run_epsilon_greedy
        procA = Process(target=funcA, args=(num_actions, num_steps))
        procB = Process(target=funcB, args=(num_actions, num_steps))
        procA.start()
        processes.append(procA)
        procB.start()
        processes.append(procB)
    for proc in processes:
        proc.join()

    print(f"Average reward when exploration is done : {results_epsilon_greedy}")
    print(f"Average reward when NO exploration : {results_greedy}")
    

if __name__ == '__main__':
    # Run the test in this mode (development / active-debug sess)
    # test_run_parallel()
    run_experiments(100, 50, 5)