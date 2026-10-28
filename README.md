# Client Selection under Unreliable Participation

A comparison of **E3CS**, **entropic online stochastic mirror descent (OSMD)**, and **uniform random sampling** for federated learning with unreliable clients.

This project studies the client-selection problem considered by Huang et al. in *Stochastic Client Selection for Federated Learning with Volatile Clients*. The goal is to improve the number of useful client updates while retaining a minimum selection opportunity for every client, and to examine how this affects model accuracy.

## Project files

- [Experiment notebook](final-convex-code.ipynb): implementation, experiment settings, saved results, and plots.
- [Project report — LaTeX source](report/main.tex): problem formulation, algorithm details, experimental design, and comparison of the results.
- [Original paper](papers/stochastic-client-selection.pdf): the reference paper by Huang et al.
- [Report figures](report/images/): accuracy and participation plots from the completed notebook.
- [Dependencies](requirements.txt).

The original paper is included as reference material and remains the work of its authors. The project report describes our implementation and experiments.

## The problem

In each federated-learning round, the server selects a fixed number of clients. Some selected clients fail to return an update, so a selection does not necessarily produce a contribution to training.

Uniform selection gives every client the same chance, but repeatedly selects clients that are unlikely to finish. Favoring reliable clients can increase successful participation, although it may also narrow the client data represented in training. The adaptive methods learn from successful and unsuccessful returns while respecting a lower bound on each client's selection probability.

For client inclusion probabilities $p_{i,t}$, the constraints are

$$
\sum_{i=1}^{K}p_{i,t}=k,\qquad \sigma_t\leq p_{i,t}\leq 1.
$$

Here, $K$ is the number of clients, $k$ is the number selected per round, and $\sigma_t$ is the fairness floor. These are inclusion probabilities for a set of clients, so they sum to $k$. The fairness constraint concerns selection opportunities; it does not guarantee equal successful contributions or equal client accuracy.

## Methods

| Method | Selection rule |
| :--- | :--- |
| E3CS | Increases weights using importance-weighted successful returns, adds a fairness allocation, and caps inclusion probabilities at one. |
| OSMD | Penalizes importance-weighted failures, then projects the updated probabilities onto the feasible set using negative entropy. |
| Random | Selects a uniform subset of distinct clients without learning from feedback. |

The adaptive methods use **pivotal sampling** by default. It selects exactly $k$ clients and preserves the prescribed inclusion probabilities needed for importance-weighted feedback. This differs from ordinary weighted sampling without replacement.

OSMD is an established method. This project adapts it to failure-based, fixed-size client selection and compares it with E3CS; it does not claim a new mirror-descent algorithm.

## Completed experiment

| Setting | Value |
| :--- | :--- |
| Dataset | EMNIST-Letters, 26 classes |
| Client data | Non-IID; 80% primary-label examples |
| Clients | 100, with 20 selected per round |
| Training budget | 400 communication rounds |
| Data per client | 450 training and 50 test examples |
| Reliability | 25 clients at each success probability: 0.1, 0.3, 0.6, and 0.9 |
| Local training | FedAvg; SGD with learning rate 0.01, momentum 0.9, and batch size 40 |
| Local epochs | Sampled from 1–4 once per client and held fixed |
| Selection base rate | 0.5; OSMD uses the matched-rate option |
| Seeds | 0, 1, and 2 |

Fixed schedule labels `0`, `0.5`, and `0.8` correspond to probability floors of 0, 0.10, and 0.16. The `inc` schedule uses no floor for the first 100 rounds, then sets the floor to 0.20, forcing uniform inclusion probabilities.

Each seed shares client data, initialization, local epoch counts, and potential participation outcomes across methods. Eight adaptive configurations and one random baseline give **27 completed runs**. Random is evaluated once per seed because its policy is independent of the fairness schedule.

Training and test pools are separated globally before constructing client datasets. Aggregation retains the reference method's fixed client weights: each successful model difference receives weight $1/K$, without renormalizing over successful clients.

## Results

**Cumulative effective participation (CEP)** counts successful selected clients over all rounds. Accuracy and CEP below are means ± sample standard deviations over three seeds. Round@75 is the mean first round reaching 75% test accuracy.

| Method | Schedule | Final accuracy (%) | CEP | Round@75 |
| :--- | :--- | ---: | ---: | ---: |
| Random | Uniform | 81.780 ± 1.054 | 3836.3 ± 17.1 | 249.3 |
| E3CS | 0 | 82.640 ± 1.115 | 6422.7 ± 104.0 | 185.0 |
| OSMD | 0 | **83.147 ± 1.093** | **6703.3 ± 19.6** | **174.3** |
| E3CS | 0.5 | 82.440 ± 1.561 | 5208.0 ± 47.8 | 228.7 |
| OSMD | 0.5 | 82.973 ± 1.436 | 5221.7 ± 33.5 | 211.3 |
| E3CS | 0.8 | 82.120 ± 1.593 | 4251.3 ± 19.3 | 245.0 |
| OSMD | 0.8 | 82.287 ± 1.171 | 4370.3 ± 8.4 | 241.0 |
| E3CS | inc | 82.527 ± 0.991 | 4226.3 ± 54.6 | 220.7 |
| OSMD | inc | 82.947 ± 1.035 | 4310.7 ± 41.0 | 201.3 |

Without a fairness floor, OSMD improves mean final accuracy by **0.507 percentage points** over E3CS and produces **4.37% more successful returns**. OSMD also has higher mean final accuracy under the remaining schedules, although individual runs do not always favor it. All runs reach 75% accuracy; none reaches 85% within 400 rounds.

These results describe one dataset and stationary client reliability over three seeds. They do not establish statistical significance or performance under changing reliability. The experiment also differs from the original paper in data splitting and sampling, so its numbers are not an exact reproduction of the published results.

### Accuracy

![Test accuracy across communication rounds](report/images/emnist_noniid_fedavg_accuracy.png)

### Effective participation

![Cumulative effective participation](report/images/emnist_noniid_fedavg_cep.png)

Lines show means across seeds; shaded bands show one sample standard deviation.

## Running the notebook

Install the dependencies in a Python environment:

```bash
python -m pip install -r requirements.txt
```

Open [final-convex-code.ipynb](final-convex-code.ipynb) in Jupyter, VS Code, or Google Colab and run the cells in order. For local Jupyter use, install it if needed:

```bash
python -m pip install jupyterlab
python -m jupyterlab
```

The notebook selects CUDA when available and otherwise runs on the CPU. Datasets are downloaded on first use. Full experiments can take several hours, depending on the hardware.

### Experiment parameters

You can change the parameters in `NOTEBOOK_ARGS` to choose the experiment:

```python
NOTEBOOK_ARGS = [
    "--datasets", "emnist",
    "--distributions", "noniid",
    "--updates", "fedavg",
    "--schedules", "0", "0.5", "0.8", "inc",
    "--seeds", "0", "1", "2",
    "--output-dir", "./results",
    "--show-plots",
]

history_df, summary_df = main(NOTEBOOK_ARGS)
```

Leave `--rounds` unset for 400 EMNIST rounds or 2500 CIFAR-10 rounds. For a short execution check, use:

```python
NOTEBOOK_ARGS = [
    "--datasets", "emnist",
    "--distributions", "noniid",
    "--updates", "fedavg",
    "--schedules", "0",
    "--seeds", "0",
    "--rounds", "2",
    "--output-dir", "./results-check",
    "--show-plots",
]

history_df, summary_df = main(NOTEBOOK_ARGS)
```

A two-round run checks execution; it does not reproduce the reported results. The code also supports CIFAR-10, IID data, and FedProx, but these settings are not included in the completed comparison above.

### Generated outputs

The output folder contains:

- `history.csv`: round-level accuracy, successful returns, and CEP.
- `summary.csv`: final metrics and accuracy-target rounds for each run.
- Separate accuracy and CEP plots for each dataset/distribution/update combination.

Results are saved after each completed run. Use a different `--output-dir` when retaining another experiment; rerunning into the same folder can overwrite saved files. Missing accuracy-target entries mean the target was not reached.

## Building the report

Compile [report/main.tex](report/main.tex) with pdfLaTeX, keeping the `images` folder beside it. Compile twice to resolve citations and cross-references. The bibliography is included in the source; no separate BibTeX step is needed. The same folder can be uploaded to Overleaf with `main.tex` selected as the main document.

## References

- Huang et al., [Stochastic Client Selection for Federated Learning with Volatile Clients](https://arxiv.org/abs/2011.08756).
- Lattimore and Szepesvári, *Bandit Algorithms*, Cambridge University Press, 2020.
- Zhao et al., [Adaptive Client Sampling in Federated Learning via Online Learning with Bandit Feedback](https://www.jmlr.org/papers/v26/24-0385.html), JMLR, 2025.
