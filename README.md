**Updates**
- Tue Sep 10 20:14:30 PDT 2024: Reran experiment + figure after bug fix


# SeqSort

Sequential Halving is a simple yet powerful multi-armed bandit algorithm used in SOTA RL algorithms such as [Gumbel MuZero](https://arxiv.org/pdf/2403.00564) and [EfficientZero V2](https://arxiv.org/pdf/2403.00564).

And it turns out SH can be slightly improved with a simple 1-line code change!

<p align="center">
  <img src="https://github.com/user-attachments/assets/3a060240-e9dd-47d6-970f-92678e2344e1" alt="Results" width="900">
</p>

Sequential Halving works by allocating a fixed budget of samples across a sequence of sampling rounds. In round one, every arm is sampled. After each round, the worst half of the arms are eliminated. This process continues until you have only one arm (the best arm estimate)[1].

<p align="center">
  <img src="https://github.com/user-attachments/assets/22bb1ac0-7850-4cd3-974a-48d40f8bd1e0" alt="Results" width="450">
</p>

SeqHalving is great because it performs well across many scenarios[1]. Also, it's easy to use: there aren't any hyperparameters to tune when using it.
<p align="center">
  <img src="https://github.com/user-attachments/assets/40f769cd-d5b8-4437-bd4b-5e322e55e13a" alt="Results" width="500">
</p>

It turns out that Sequential Halving can be improved with a simple change: Rerank *all* the arms between rounds without doing any elimination. The sampling round size is still halved at the end of each round, but all arms are eligible to participate in subsequent rounds.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8ae403f4-f8ee-4420-a1c2-386e62ca6fb2" alt="Results" width="600">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/1965a935-68fa-4431-bb17-46aa1a571a51" alt="Results" width="600">
</p>

Doing so increases the chances of finding the true best arm. I doubt this is new/novel, and it's only a slight improvement. But still, it's useful to know!

Refs:  
[1] [Almost Optimal Exploration in Multi-Armed Bandits](https://proceedings.mlr.press/v28/karnin13.pdf)  
