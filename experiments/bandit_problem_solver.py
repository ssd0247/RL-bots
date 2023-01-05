import numpy as np
import time
from scipy.stats import beta
import matplotlib # noqa
matplotlib.use('Agg') # noqa
import matplotlib.pyplot as plt

class Bandit:
    def generate_reward(self, i):
        raise NotImplementedError
    

class BernoulliBandit(Bandit):
    def __init__(self, n, probabs=None):
        assert probabs is None or len(probabs) != n
        self.n = n
        if probabs is None:
            np.random.seed(int(time.time()))
            self.probabs = [np.random.random() for _ in range(self.n)]
        else:
            self.probabs = probabs
        
        self.best_probabs = max(self.probabs)
    
    def generate_reward(self, i):
        # The player selected the i-th machine.
        if np.random.random() < self.probabs[i]:
            return 1
        else:
            return 0

class Solver:
    def __init__(self, bandit):
        assert isinstance(bandit, BernoulliBandit)
        np.random.seed(int(time.time()))
        self.bandit = bandit
        self.counts = [0] * self.bandit.n
        self.actions = [] # A list of machine ids, 0 to bandit n-1.
        self.regret = 0. # cummulative regret.
        self.regrets = [0.] # History of cummulative regrets
    
    def update_regret(self, i):
        # i (int) : index of the selected machine.
        self.regret += self.bandit.best_probabs - self.bandit.probabs[i]
        self.regrets.append(self.regret)
    
    @property
    def estimated_probabs(self):
        raise NotImplementedError
    
    def run_one_step(self):
        """Return the machine index to take action on."""
        raise NotImplementedError
    
    def run(self, num_steps):
        assert self.bandit is not None
        for _ in range(num_steps):
            i = self.run_one_step()
            self.counts[i] += 1
            self.actions.append(i)
            self.update_regret(i)

class EpsilonGreedy(Solver):
    def __init__(self, bandit, eps, init_probab=1.0):
        """
        eps (float) : the probability to explore at each time step.
        init_probs (float) : default to be 1.0; optimistic initialization
        """
        super(EpsilonGreedy, self).__init__(bandit)
        assert 0. <= eps <= 1.0
        self.eps = eps
        self.estimates = [init_probab] * self.bandit.n # Optimistic weighted
    
    @property
    def estimated_probabs(self):
        return self.estimates
    
    def run_one_step(self):
        if np.random.random() < self.eps:
            # Let's do random exploration !
            i = np.random.randint(0, self.bandit.n)
        else:
            # Pick the best one.
            i = max(range(self.bandit.n), key=lambda x: self.estimates[x])

        r = self.bandit.generate_reward(i)
        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])
        return i

class BayesianUCB(Solver):
    """Assuming Beta prior."""
    def __init__(self, bandit, c=3, init_a=1, init_b=1):
        """
        c (float) : how many standard dev to consider as an upper confidence bound.
        init_a (int) : initial value of a in Beta(a, b).
        init_b (int) : initial value of b in Beta(a, b).
        """
        super(BayesianUCB, self).__init__(bandit)
        self.c = c
        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n
    
    @property
    def estimated_probabs(self):
        return [self._as[i] / float(self._as[i] + self._bs[i]) for i in range(self.bandit.n)]
    
    def run_one_step(self):
        # Pick the best one with consideration of upper confidence bounds.
        i = max(
            range(self.bandit.n),
            key=lambda x: self._as[x] / float(self._as[x] + self._bs[x] + beta.std(
                self._as[x], self._bs[x] * self.c
            )))
        r = self.bandit.generate_reward(i)

        # Updat the Gaussian posterior
        self._as[i] += r
        self._bs[i] += (1 - r)

        return i

class ThomsonSampling(Solver):
    def __init__(self, bandit, init_a=1, init_b=1):
        """
        init_a (int) : initial value of a in Beta(a, b).
        init_b (int) : initial value of b in Beta(a, b).
        """
        super(ThomsonSampling, self).__init__(bandit)
        self._as = [init_a] * self.bandit.n
        self._bs = [init_b] * self.bandit.n

    @property
    def estimated_probabs(self):
        return [self._as[i] / (self._as[i] + self._bs[i]) for i in range(self.bandit.n)]
    
    def run_one_step(self):
        samples = [np.random.beta(self._as[x], self._bs[x]) for x in range(self.bandit.n)]
        i = max(range(self.bandit.n), key=lambda x: samples[x])
        r = self.bandit.generate_reward(i)
        self._as[i] += r
        self._bs[i] += (1 - r)
        return i

class UCB1(Solver):
    def __init__(self, bandit, init_probab=1.0):
        super(UCB1, self).__init__(bandit)
        self.t = 0
        self.estimates = [init_probab] * self.bandit.n
    
    @property
    def estimated_probabs(self):
        return self.estimates
    
    def run_one_step(self):
        self.t += 1

        # Pick the best one with the consideration of upper confidence bounds.
        i = max(range(self.bandit.n), key=lambda x: self.estimates[x] + np.sqrt(
            2 * np.log(self.t) / (1 + self.counts[x])))
        r = self.bandit.generate_reward(i)
        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])
        return i

def plot_results(solvers, solver_names, figname):
    """
    Plot the results by multi-armed bandit solvers.

    Args :
        solvers (list<Solver>): All of them should have been fitted.
        solver_names (list<str>)
        figname (str)
    """
    assert len(solvers) == len(solver_names)
    assert all(map(lambda s: isinstance(s, Solver), solvers))
    assert all(map(lambda s: len(s.regrets) > 0, solvers))

    b = solvers[0].bandit

    fig = plt.figure(figsize=(14, 4))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot()

    # Sub.fig. 1 : Regrets in time.
    for i, s in enumerate(solvers):
        ax1.plot(range(len(s.regrets)), s.regrets, label=solver_names[i])
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Cumulative Regret')
    ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax1.grid('k', ls='--', alpha=0.3)

    # Sub.fig. 2 : Probabilities estimated by solvers.
    sorted_indices = sorted(range(b.n), key=lambda x: b.probabs[x])
    ax2.plot(range(b.n), [b.probabs[x] for x in sorted_indices], 'k--', markersize=12)
    for s in solvers:
        ax2.plot(range(b.n), [s.estimated_probabs[x] for x in sorted_indices], 'x', markeredgewidth=2)
    ax2.set_xlabel('Actions sorted by ' + r'$\theta$')
    ax2.set_ylabel('Estimated')
    ax2.grid('k', ls='--', alpha=0.3)

    # Sub.fig. 3 : Action counts
    for s in solvers:
        ax3.plot(range(b.n), np.array(s.counts) / float(len(solvers[0].regrets)), ls='--', lw=2)
    ax3.set_xlabel('Actions')
    ax3.set_ylabel('Frac. # trials')
    ax3.grid('k', ls='--', alpha=0.3)

    plt.savefig(figname)

def experiment(K, N):
    """
    Run a small experiment on solving a Bernoulli bandit with K slot machines,
    each with a randomly initialized reward probability.

    Args:
        K (int) : number of slot machines
        N (int) : number of time steps to try.
    """

    b = BernoulliBandit(K)
    print("Randomly generated Bernoulli bandit has reward probabilities:\n", b.probabs)
    print("The best machine has index : {} and probab : {}".format(
        max(range(K), key=lambda i: b.probabs[i]), max(b.probabs)))
    
    test_solvers = [
        # EpsilonGreedy(b, 0)
        # EpsilonGreedy(b, 1)
        EpsilonGreedy(b, 0.01),
        UCB1(b),
        BayesianUCB(b, 3, 1, 1),
        ThomsonSampling(b, 1, 1)
    ]

    names = [
        # 'Full-exploitation',
        # 'Full-exploitation',
        r'$\epsilon$' + '-Greedy',
        'UCB1',
        'Bayesian UCB',
        'Thomson Sampling'
    ]

    for s in test_solvers:
        s.run(N)
    
    plot_results(test_solvers, names, "results_K{}_N{}.png".format(K, N))

if __name__ == '__main__':
    experiment(10, 5000)