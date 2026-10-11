# Intelligent Systems · Selected Algorithms

Four small, standalone Python examples from university work on rule-based agents, neural learning, local search and evolutionary optimization.

## Examples

| File | Demonstrates |
|---|---|
| `examples/wall_follow_agent.py` | Stateful rule-based decisions from eight binary proximity sensors |
| `examples/xor_network.py` | Forward propagation and batch backpropagation implemented with NumPy |
| `examples/tsp_local_search.py` | Hill climbing and threshold acceptance for the Traveling Salesperson Problem |
| `examples/n_queens_evolution.py` | Permutation encoding, ordered crossover, swap mutation and tournament selection |

## Run

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python examples/xor_network.py
python examples/tsp_local_search.py
python examples/n_queens_evolution.py
python examples/wall_follow_agent.py
```

Each file has a deterministic demonstration and lightweight assertions, so it can be reviewed without instructor packages, notebooks or private course datasets.

## Background

The original workspace covered seven assignments. For GitHub, it was reduced to the four complete examples above; assignment sheets, starter templates, instructor support code, course datasets and unfinished exercises are not included.

The source coursework included both individual work and explicitly marked group exercises by **Aziz Zarrouki, Jihed Khattat and Rayan Brigui**. The examples here preserve that team context and do not use repository history to make unsupported individual-authorship claims. University coursework publication is permitted; no course grade is claimed.
