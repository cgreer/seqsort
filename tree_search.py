'''
How does SeqSort Compare to SeqHalving on a Typical Gumbel AlphaZero Setup?

Gumbel AlphaZero select ~16 candidate actions and then invests ~64 simulations
during search to update the prior policy. Does SeqSort perform well in this
type of situation?


[Questions]

1. Does SeqSort do worse than SeqHalving?
    - NO. Effectively the same, perhaps very slightly better.

2. Does re-ranking by approx posterior perform better?
    - NO.


[Results]

Train Conditions
Experiment                       E[Error] Std[Error]                                                                            E[Acc]   Std[Acc]
SeqHalving-16-64                  0.23813    0.00049           [                    |--◇--|  |                            ]    0.75697    0.00096           [                     |--◇--| |                            ]
SeqHalvingBayes-16-64             0.23787    0.00049           [                   |--◇--|   |                            ]    0.75848    0.00096           [                        |-◇--|                            ]
SeqSort-16-64                     0.23593    0.00050           [             |--◇--|         |                            ]    0.76060    0.00095           [                          |--|--|                         ]
SeqSortBayes-16-64                0.23669    0.00050           [                |--◇-|       |                            ]    0.75857    0.00097           [                        |-◇--|                            ]

Test Conditions
SeqHalving-32-900                 0.24512    0.00042           [                          |--|-|                          ]    0.67599    0.00105           [                           |-|◇--|                        ]
SeqHalvingBayes-32-900            0.24503    0.00041           [                          |--|-|                          ]    0.67583    0.00104           [                           |-|◇--|                        ]
SeqSort-32-900                    0.24392    0.00040           [                       |-◇--||                            ]    0.67678    0.00101           [                            ||-◇--|                       ]
SeqSortBayes-32-900               0.24444    0.00042           [                        |--◇-|                            ]    0.67654    0.00106           [                            ||-◇--|                       ]

'''
from dataclasses import dataclass
from typing import Callable, List
import math
from random import random, shuffle, randint

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

    Note: Doesn't always use full budget + handle when not enough
    samples compared to number of arms.
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


def round_allocations_gaz(m: int, n: int):
    '''
    :m - num candidates
    :n - num simulations

    Allocator for Gumbel AZ.

    Aspires to:
        (a) distribute n evenly across rounds and
        (b) give every arm at least 1 sample each round.

    When it isn't possible to give each arm a sample, it will randomly
    distrubute the remaining budget to the candidates in that round.
    Any rounding leftovers are redistributed to round 1.
    '''
    if m == 1:
        rounds = [[n]]
        return rounds

    n_rounds = math.ceil(math.log2(m))
    round_target = n / n_rounds

    budget = n
    rounds = []
    n_cands = m
    for r_i in range(n_rounds):
        round_desired = max(n_cands * 1, round_target)
        round_budget = min(budget, round_desired)

        # Enough samples
        # - Distribute uniformly
        if (round_budget / n_cands) >= 1.0:
            n_per = int(round_budget // n_cands)
            allocs = [n_per] * n_cands
            budget -= (n_per * n_cands)

        # Not enough remaining samples to go around
        # - Randomly distribute
        else:
            cand_idxs = list(range(n_cands))
            shuffle(cand_idxs)
            allocs = [0] * n_cands
            for i in range(int(round_budget)):
                allocs[i] = 1
                budget -= 1
        rounds.append(allocs)
        n_cands = math.ceil(n_cands / 2)

    # Distribute any leftovers (due to rounding) to round 1
    assert budget >= 0
    while budget > 0:
        rounds[0][randint(0, m-1)] += 1
        budget -= 1

    return rounds


def range_bar(low, mid, high, width=60, lb=-1.0, hb=1.0):
    '''
    [              |--------◇-----|--|                         ]
    '''
    brange = hb - lb
    low_off = (low - lb) / brange
    low_off = int(low_off * width)
    mid_off = (mid - lb) / brange
    mid_off = int(mid_off * width)
    high_off = (high - lb) / brange
    high_off = int(high_off * width)
    center_off = ((lb + brange/2.0) - lb) / brange
    center_off = int(center_off * width)

    s = list("[" + (" " * (width - 2)) + "]")
    for i in range(low_off + 1, high_off):
        s[i] = "-"
    s[low_off] = "|"
    s[mid_off] = "◇"
    s[high_off] = "|"
    s[center_off] = "|"
    s = "".join(s)
    return s


@dataclass
class ClippedNormArm:
    '''
    Intended to approximate sampling advantage values (e.g. -5.0 to 5.0)
    from a GAZ-style tree search.
    '''
    mean: float
    std: float
    bounds: List[float]

    @classmethod
    def generate(Cls, mean_range, std_range, bounds):
        mean = mean_range[0] + random() * (mean_range[1] - mean_range[0])
        std = std_range[0] + random() * (std_range[1] - std_range[0])
        return Cls(
            mean=mean,
            std=std,
            bounds=bounds,
        )

    def sample(self, N):
        samps = np.random.normal(self.mean, self.std, N)
        samps[samps < self.bounds[0]] = self.bounds[0]
        samps[samps > self.bounds[1]] = self.bounds[1]
        return samps


NumSamples = int
RewardSum = float


@dataclass
class ResultStats:
    mean: float
    std: float
    range: List[float] # percentiles: [2.5, 50.0, 97.5]

    @classmethod
    def new(Cls):
        return Cls(0.0, 0.0, [0.0, 0.0, 0.0])


class Techniques:
    SEQ_HALVING = 0
    SEQ_SORT = 1
    SEQ_HALVING_BAYES = 2
    SEQ_SORT_BAYES = 3


@dataclass
class Experiment:
    m: int # n of arms
    n: int # n of samples (i.e. "budget")
    technique: Techniques
    errors: List[float] # absolute errors between true best mean and chosen mean
    error_stats: ResultStats
    corrects: List[int] # 1 if chose correct arm else 0
    acc_stats: ResultStats


@dataclass
class Candidate:
    id: int
    sample: Callable[[NumSamples], RewardSum] # fxn to sample from bandit "arm"
    n_samples: int = 0
    reward_sum: float = 0.0
    reward_avg: float = 0.0


def run_experiments(
    setups,
    n_replicates: int,
    mean_range: List[float],
    std_range: List[float],
    bounds: List[float],
    n_resims,
):
    n_0 = 1 # prior strength; i.e. num of psuedosamples

    # Collect data
    for setup in setups:
        for rep_i in range(n_replicates):
            # Generate arms
            arms = []
            for _ in range(setup.m):
                arm = ClippedNormArm.generate(
                    mean_range=mean_range,
                    std_range=std_range,
                    bounds=bounds,
                )
                arms.append(arm)
            best_arm = np.argmax([x.mean for x in arms])
            best_mean = arms[best_arm].mean

            # Run bandit for setup
            cands = [Candidate(id=i, sample=arms[i].sample) for i in range(setup.m)]
            for sh_round in round_allocations_gaz(setup.m, setup.n):
                for cand_idx, n_samples in enumerate(sh_round):
                    cand = cands[cand_idx]
                    cand.n_samples += n_samples
                    cand.reward_sum += cand.sample(n_samples).sum()
                    cand.reward_avg = cand.reward_sum / cand.n_samples

                # Halve/Sort
                if setup.technique == Techniques.SEQ_HALVING:
                    cands[:len(sh_round)] = sorted(cands[:len(sh_round)], key=lambda x: x.reward_avg, reverse=True)
                elif setup.technique == Techniques.SEQ_SORT:
                    cands.sort(key=lambda x: x.reward_avg, reverse=True)
                elif setup.technique == Techniques.SEQ_HALVING_BAYES:
                    # Calc prior (emperical mean)
                    # XXX: Only use arms with visits for u_o?
                    # u_post = (x.n_samples*x.reward_avg + n_0*u_0) / (x.n_samples + n_0)
                    u_0 = sum([x.reward_avg for x in cands]) / len(cands)
                    cands[:len(sh_round)] = sorted(
                        cands[:len(sh_round)],
                        key=lambda x: (x.n_samples*x.reward_avg + n_0*u_0) / (x.n_samples + n_0),
                        reverse=True,
                    )
                elif setup.technique == Techniques.SEQ_SORT_BAYES:
                    # Calc prior (emperical mean)
                    # u_post = (x.n_samples*x.reward_avg + n_0*u_0) / (x.n_samples + n_0)
                    u_0 = sum([x.reward_avg for x in cands]) / len(cands)
                    cands.sort(
                        key=lambda x: (x.n_samples*x.reward_avg + n_0*u_0) / (x.n_samples + n_0),
                        reverse=True,
                    )
                else:
                    raise KeyError()

            best_cand = cands[0]

            # Record results
            setup.errors.append(abs(best_mean - best_cand.reward_avg))
            setup.corrects.append(1 if best_cand.id == best_arm else 0)

    # Compute statistics
    for setup in setups:
        for data, stats in (
            (setup.errors, setup.error_stats),
            (setup.corrects, setup.acc_stats),
        ):
            stats.mean = np.mean(data)
            boot = np.random.choice(
                data,
                size=(n_resims, len(data)),
                replace=True,
            ).mean(1)
            stats.std = boot.std()
            stats.range = np.percentile(boot, q=[2.5, 50.0, 97.5])


if __name__ == "__main__":
    np.random.seed(42)

    setups = [
        # Train settings: 16/64
        Experiment(
            m=16,
            n=64,
            technique=Techniques.SEQ_HALVING,
            errors=[],
            error_stats=ResultStats.new(),
            corrects=[],
            acc_stats=ResultStats.new(),
        ),
        Experiment(
            m=16,
            n=64,
            technique=Techniques.SEQ_SORT,
            errors=[],
            error_stats=ResultStats.new(),
            corrects=[],
            acc_stats=ResultStats.new(),
        ),
        Experiment(
            m=16,
            n=64,
            technique=Techniques.SEQ_HALVING_BAYES,
            errors=[],
            error_stats=ResultStats.new(),
            corrects=[],
            acc_stats=ResultStats.new(),
        ),
        Experiment(
            m=16,
            n=64,
            technique=Techniques.SEQ_SORT_BAYES,
            errors=[],
            error_stats=ResultStats.new(),
            corrects=[],
            acc_stats=ResultStats.new(),
        ),

        # Eval settings: 32/900
        Experiment(
            m=32,
            n=900,
            technique=Techniques.SEQ_HALVING,
            errors=[],
            error_stats=ResultStats.new(),
            corrects=[],
            acc_stats=ResultStats.new(),
        ),
        Experiment(
            m=32,
            n=900,
            technique=Techniques.SEQ_SORT,
            errors=[],
            error_stats=ResultStats.new(),
            corrects=[],
            acc_stats=ResultStats.new(),
        ),
        Experiment(
            m=32,
            n=900,
            technique=Techniques.SEQ_HALVING_BAYES,
            errors=[],
            error_stats=ResultStats.new(),
            corrects=[],
            acc_stats=ResultStats.new(),
        ),
        Experiment(
            m=32,
            n=900,
            technique=Techniques.SEQ_SORT_BAYES,
            errors=[],
            error_stats=ResultStats.new(),
            corrects=[],
            acc_stats=ResultStats.new(),
        ),

    ]

    # Collect data
    run_experiments(
        setups=setups,
        n_replicates=200_000, # Hope your computer has as much memory as mine!
        mean_range=[-5.0, +5.0],
        std_range=[+0.1, +2.0],
        bounds=[-5.0, +5.0],
        n_resims=1000, # For bootstrap
    )

    # Display results
    range_params = [
        # Train
        [0.24, 0.01, 0.76, 0.02],
        [0.24, 0.01, 0.76, 0.02],
        [0.24, 0.01, 0.76, 0.02],
        [0.24, 0.01, 0.76, 0.02],

        # Eval
        [0.245, 0.01, 0.675, 0.02],
        [0.245, 0.01, 0.675, 0.02],
        [0.245, 0.01, 0.675, 0.02],
        [0.245, 0.01, 0.675, 0.02],
    ]
    print(
        "Experiment".ljust(30),

        "E[Error]".rjust(10),
        "Std[Error]".rjust(10),
        "".ljust(70),

        "E[Acc]".rjust(10),
        "Std[Acc]".rjust(10),
        "".ljust(70),
    )
    for exp_i, exp in enu(setups):
        tech_name = {
            0: "SeqHalving",
            1: "SeqSort",
            2: "SeqHalvingBayes",
            3: "SeqSortBayes",
        }[exp.technique]
        name = f"{tech_name}-{exp.m}-{exp.n}"

        rng_mean, pm = range_params[exp_i][0], range_params[exp_i][1]
        rng_bounds = [rng_mean - pm, rng_mean + pm]
        error_range = range_bar(*exp.error_stats.range, width=60, lb=rng_bounds[0], hb=rng_bounds[1])

        rng_mean, pm = range_params[exp_i][2], range_params[exp_i][3]
        rng_bounds = [rng_mean - pm, rng_mean + pm]
        acc_range = range_bar(*exp.acc_stats.range, width=60, lb=rng_bounds[0], hb=rng_bounds[1])
        print(
            name.ljust(30),

            f"{exp.error_stats.mean:.05f}".rjust(10),
            f"{exp.error_stats.std:.05f}".rjust(10),
            f"{error_range}".rjust(70),

            f"{exp.acc_stats.mean:.05f}".rjust(10),
            f"{exp.acc_stats.std:.05f}".rjust(10),
            f"{acc_range}".rjust(70),
        )
