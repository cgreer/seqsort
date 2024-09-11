'''
Experiment demonstrating that Sequential Sorting is an improvement over Sequential Halving

Reproduces experiments shown in Figure 1 from "Almost Optimal Exploration in Multi-Armed Bandits" (https://proceedings.mlr.press/v28/karnin13.pdf)
'''
from dataclasses import dataclass
from typing import Callable, List
import math
import pprint

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

import numpy as np

ceil = math.ceil
floor = math.floor
log2 = math.log2
sqrt = math.sqrt

enu = enumerate


def round_allocations(m: int, n: int) -> List[List[int]]:
    '''
    :m - num candidates
    :n - num total samples (fixed budget)

    See "Algorithm 2" in https://proceedings.mlr.press/v28/karnin13.pdf
    '''
    n_rounds = ceil(log2(m))
    budget = n
    rounds = []
    round_size = m # How many candidates get sampled from this round?
    for r_i in range(n_rounds):
        t_r = floor(n / (round_size * ceil(log2(m))))
        allocations = [t_r] * round_size
        rounds.append(allocations)
        budget -= (t_r * round_size)
        round_size = ceil(round_size / 2)
    assert budget >= 0
    return rounds


@dataclass
class BernoulliArm:
    p: float

    def sample(self, N):
        return np.random.binomial(1, self.p, N)


##################
# Setups (Distribution of Bernoulli candidates)
#################

def setup_1_ps(n):
    ps = [0.5] # optimal
    ps += ([0.45] * (n - 1))
    assert len(ps) == n
    return ps


def setup_2_ps(n):
    ps = [0.5] # optimal
    for i in range(2, n + 1): # 1-based
        if i <= (ceil(sqrt(n)) + 1):
            ps.append(0.5 - (1/(2*n)))
        else:
            ps.append(0.45)
    assert len(ps) == n
    return ps


def setup_3_ps(n):
    ps = [0.5] # optimal
    for i in range(2, n + 1): # 1-based
        if i <= 6:
            ps.append(0.5 - (1/(5*n)))
        elif i <= (6 + (2 * (ceil(sqrt(n))))):
            ps.append(0.49)
        else:
            ps.append(0.35)
    assert len(ps) == n
    return ps


def setup_4_ps(n):
    ps = [0.5] # optimal
    other = list(np.linspace(0.50 - (1.0 / (5.0 * n)), 0.25, n-1))
    ps.extend(other)
    assert len(ps) == n
    return ps


def setup_5_ps(n):
    ps = [0.5] # optimal
    p2 = 0.5 - (1 / (5 * n))
    ps.append(p2)
    if n == 20:
        r = 0.9635
    elif n == 40:
        r = 0.982
    elif n == 80:
        r = 0.9915
    else:
        raise KeyError()
    for i in range(2, n):
        ps.append(p2 * (r ** (i - 1)))
    assert len(ps) == n
    return ps


def setup_6_ps(n):
    ps = [0.5] # optimal
    ps.append(0.5 - (1/(10*n)))
    ps += ([0.40] * (n - 2))
    assert len(ps) == n
    return ps


SETUP_NAMES = [
    "One group of suboptimal arms",
    "Two groups of suboptimal arms",
    "Three groups of suboptimal arms",
    "Arithmetic",
    "Geometric",
    "One real competitor",
]
SETUP_R_MULT = [8, 5, 1, 1, 1, 1] # multiplier for replicates
SETUP_PS = [setup_1_ps, setup_2_ps, setup_3_ps, setup_4_ps, setup_5_ps, setup_6_ps]
SETUP_BUDGETS = [ # S x (Arms, Budget)]
    [(20, 7600),   (40, 15600),  (80, 31600)], # noqa Setup 1
    [(20, 13599),  (40, 57599),  (80, 258400)], # noqa Setup 2
    [(20, 150177), (40, 340888), (80, 982488)], # noqa Setup 3
    [(20, 14005),  (40, 57430),  (80, 232579)], # noqa Setup 4
    [(20, 33220),  (40, 214888), (80, 1436445)], # noqa Setup 5
    [(20, 41799),  (40, 163799), (80, 647800)], # noqa Setup 6
]

NumSamples = int
RewardSum = float


@dataclass
class Candidate:
    id: int
    sample: Callable[[NumSamples], RewardSum] # fxn to sample from bandit "arm"
    n_samples: int = 0
    reward_sum: float = 0.0
    reward_avg: float = 0.0


def run_experiment(n_replicates: int, n_resims: int = 1000):
    '''
    :n_replicates - number of replicates
    :n_resims - num of times you resim for bootstrapping confidence
    '''
    figure_data = [] # [ ["SeqHalving", 1, [0.246, 0.26517, 0.29022]], ... ]
    for setup, setup_name in enu(SETUP_NAMES):
        print(f"\nSetup {setup + 1}")
        results = [
            ["SeqHalving", setup + 1, [0.0, 0.0, 0.0]],
            ["SeqSort", setup + 1, [0.0, 0.0, 0.0]],
        ]
        for sub_setup in (0, 1, 2): # 20, 40, 80 arms
            n_cands, budget = SETUP_BUDGETS[setup][sub_setup]
            ps = SETUP_PS[setup](n_cands)
            arms = [BernoulliArm(ps[i]) for i in range(n_cands)]
            best_mean = np.max(ps)
            best_arm = np.argmax(ps)
            assert best_arm == 0

            print(setup_name, sub_setup, n_cands, budget, ps[:5], ps[-5:])
            errors = [[], []] # abs(selected arm.mean - best arm.mean)
            incorrects = [[], []] # 1 if didnt' select best arm else 0
            for _ in range(n_replicates * SETUP_R_MULT[setup]):

                #####################
                # Sequential Halving
                #####################
                cands = [Candidate(id=i, sample=arms[i].sample) for i in range(n_cands)]
                for sh_round in round_allocations(n_cands, budget):

                    # Sampling Round (size of each round is ~half of previous)
                    # - Sample :n_samples from len(sh_round) *remaining* candidates
                    # - Update the stats of candidates sampled from
                    for cand_idx, n_samples in enumerate(sh_round):
                        cand = cands[cand_idx]
                        cand.n_samples += n_samples
                        cand.reward_sum += cand.sample(n_samples).sum()
                        cand.reward_avg = cand.reward_sum / cand.n_samples

                    # Halve
                    # - Sort only the *remaining* candidates by their reward_avg descending
                    cands[:len(sh_round)] = sorted(cands[:len(sh_round)], key=lambda x: x.reward_avg, reverse=True)

                best_cand = cands[0]

                errors[0].append(abs(best_mean - best_cand.reward_avg))
                incorrects[0].append(1 if best_cand.id != best_arm else 0)

                #########################
                # Sequential Sort
                #########################
                cands = [Candidate(id=i, sample=arms[i].sample) for i in range(n_cands)]
                for sh_round in round_allocations(n_cands, budget):

                    # Sampling Round (size of each round is ~half of previous)
                    # - Sample :n_samples from len(sh_round) *current* top candidates
                    # - Update the stats of candidates sampled from
                    for cand_idx, n_samples in enumerate(sh_round):
                        cand = cands[cand_idx]
                        cand.n_samples += n_samples
                        cand.reward_sum += cand.sample(n_samples).sum()
                        cand.reward_avg = cand.reward_sum / cand.n_samples

                    # Sort
                    # - Sort *all* candidates by rew_avg descending
                    cands.sort(key=lambda x: x.reward_avg, reverse=True)

                best_cand = cands[0]

                errors[1].append(abs(best_mean - best_cand.reward_avg))
                incorrects[1].append(1 if best_cand.id != best_arm else 0)

            #####################
            # Results
            #####################
            for i in (0, 1):
                # Get error and bootstrap confidence
                mean_error = np.mean(errors[i])
                mean_bootstrap = np.random.choice(errors[i], size=(n_resims, len(errors[i])), replace=True).mean(1)
                mean_confidence = mean_bootstrap.std()

                mean_incorrect = np.mean(incorrects[i])
                incorrect_bootstrap = np.random.choice(incorrects[i], size=(n_resims, len(incorrects[i])), replace=True).mean(1)
                incorrect_confidence = incorrect_bootstrap.std()
                print(
                    results[i][0].ljust(25),
                    f"{mean_error:.05f}".ljust(10),
                    f"{mean_confidence * 2:.05f}".ljust(10),

                    f"{mean_incorrect:.05f}".ljust(10),
                    f"{incorrect_confidence:.05f}".ljust(10),
                )
                results[i][2][sub_setup] = mean_incorrect
        figure_data.extend(results)

    # Make figure
    figure(figure_data)


def figure(figure_data):
    pprint.pprint(figure_data)

    # Set font
    font = "PT Sans"
    plt.rcParams['font.family'] = font
    plt.rcParams['font.sans-serif'] = [font]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars
    x = np.arange(len(figure_data))
    width = 0.25
    gnudge = 0.05
    error_probs = [x[2] for x in figure_data]
    for i in range(len(figure_data)):
        # colors = ["lightsteelblue", "cornflowerblue", "royalblue"]
        colors = ["#d4dfd2", "#b7cab5", "#9bb597"] # Green
        if i % 2 == 0:
            # colors = ["#f5e8cc", "#efd9a9", "#e7c780"] # Yellow
            # colors = ["#f4c7ba", "#E99077", "#E16644"] # Red
            colors = ["#ee9b5e", "#f3b487", "#f8d6bc"] # Orange
            colors = ["#f8d6bc", "#f3b487", "#ee9b5e"] # Orange

        nudge = gnudge if (i % 2) == 0 else -gnudge
        ax.bar(x[i] - width + nudge, error_probs[i][0], width, label='Group 1', color=colors[0])
        ax.bar(x[i] + nudge, error_probs[i][1], width, label='Group 2', color=colors[1])
        ax.bar(x[i] + width + nudge, error_probs[i][2], width, label='Group 3', color=colors[2])

    # Titles + x/y labels
    ax.set_title('SeqSort: Improving Sequential Halving by Removing Elimination', fontsize=16, fontweight="bold")
    ax.set_ylabel('Error Probability', fontsize=14, fontweight="bold")
    ax.set_xlabel('Setup', fontsize=14, fontweight="bold")

    # Ticks / Grid
    ax.grid(axis='y', which='major', linestyle='-', linewidth=0.4, color='gray', alpha=0.3)
    ax.set_ylim(0.00, 0.40)

    ax.set_xticks(x)
    ax.set_xticklabels(["" for x in figure_data])
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', which='major', colors='white') # white to hide

    # Custom formatter to place labels on minor ticks btwn major ticks
    between_labels = [
        "Suboptimal (1)", "",
        "Suboptimal (2)", "",
        "Suboptimal (3)", "",
        "Arithmetic", "",
        "Geometric", "",
        "One Real Competitor"
    ]

    def minor_tick_formatter(x, pos):
        if x % 1 == 0.5:  # Check if it's a minor tick
            return between_labels[int(x - 0.5)]
        return ''

    ax.xaxis.set_minor_formatter(FuncFormatter(minor_tick_formatter))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))
    plt.setp(ax.xaxis.get_minorticklabels(), rotation=0, ha='center', va='top')
    minor_ticks = x[:-1] + 0.5
    ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))

    n_y_ticks = 11
    y_min, y_max = ax.get_ylim()
    ax.set_yticks(np.linspace(y_min, y_max, n_y_ticks))

    # Legend 1: Which algo
    sh_color = "#e87e31" # higher contrast
    sr_color = "#70956a"
    ax.text(0.018, 1.0 - 0.04, 'SeqHalving', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=14, fontweight='bold', color=sh_color)
    ax.text(0.018, 1.0 - 0.04 - 0.05, 'SeqSort [SeqHalving w/out elimination]', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=14, fontweight='bold', color=sr_color)

    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    run_experiment(n_replicates=20_000) # Prob overkill but death to error bars
