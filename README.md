Client Selection under Unreliable Participation

This project compares **E3CS**, **entropic online stochastic mirror descent (OSMD)**, and **uniform random sampling** for federated learning with unreliable clients. The goal is to increase successful client participation while preserving minimum selection opportunities.

## Included files

- **Code:** experiment notebook with the implementation, results, and plots.
- **Report:** problem formulation, algorithm details, and experimental comparison.
- **Original paper:** *Stochastic Client Selection for Federated Learning with Volatile Clients* by Huang et al., included as reference material.

## Running the code

Install the required packages:

```bash
pip install torch torchvision numpy pandas matplotlib
```

Open the notebook in Jupyter, VS Code, or Google Colab and run the cells in order. You can change the experiment settings in `NOTEBOOK_ARGS`. Datasets are downloaded on first use; the code uses a GPU when available and otherwise runs on the CPU.

## Experiment and results

The completed experiment uses non-IID EMNIST-Letters, FedAvg, 100 clients, 20 selections per round, 400 rounds, and three seeds.

Without a fairness floor, OSMD achieves **83.15%** mean final accuracy, compared with **82.64%** for E3CS and **81.78%** for uniform random sampling. OSMD also produces **4.37% more successful returns** than E3CS in this setting. Full results and limitations are discussed in the report.

This project adapts an established OSMD method; it does not claim a new mirror-descent algorithm. Results describe the evaluated setting and are not an exact reproduction of the original paper.

## Reference

Huang et al., [Stochastic Client Selection for Federated Learning with Volatile Clients](https://arxiv.org/abs/2011.08756).
