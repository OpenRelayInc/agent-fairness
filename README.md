# Agent Fairness

A simulator and loopback HTTP router for evaluating workflow-level fairness in LLM
inference. Compare immediate admission with capacity rejection, first-in first-out
(FIFO) queueing, and fairness-based queueing on the same seeded workload.

This is OpenRelay’s router-only experiment inspired by [Towards Fair Resource
Sharing in Agentic LLM Serving](https://arxiv.org/abs/2611.23558) (Sehyeon Lee et
al., 2026). It approximates the paper’s fairness score at request boundaries.
Duru also schedules waiting requests inside the engine, makes admission exceptions
under cache pressure, and scales GPU capacity.

## Run the simulator

Use Python 3.12. The simulator needs only the standard library.

```sh
python3 simulate.py --scenario mixed --seed 42 --out results/simulated
```

The output includes a JSON summary and request/workflow CSVs for each policy.
The mixed scenario tests 144 synthetic workflows with causal forks and joins.
Read all assigned workflow outcomes alongside the slowdown and queue-wait
percentiles, which summarize successful completions. In the mixed scenario at
seed 42, both queueing policies complete all 144 workflows. Fairness improves
p99 normalized slowdown while increasing p99 request wait. These timings come
from the simulated engine.

## Run the loopback tests

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -W error -m unittest discover -p 'test_*.py' -v
```

## Replay against your own engine

Start a private vLLM-compatible engine on a numeric loopback address. Set
`--served-model` to the model name accepted by that engine.

```sh
.venv/bin/python router.py --upstream http://127.0.0.1:8001 --port 8088 \
  --served-model local-model --capacity 8 --mode fair
```

In a second terminal:

```sh
.venv/bin/python replay.py --url http://127.0.0.1:8088 \
  --scenario light --time-scale 0.01 --out results/http-fair
```

The router’s listener is local. Keep it on loopback: workflow IDs and cost
estimates are supplied by the test client. The replay uses fictional prompts and
tool results and drives the streaming Chat Completions API.

See [the experiment guide](EXPERIMENT.md) for the scoring model, measured
standalone calibration, three-policy GPU comparison, and how to interpret the
results.

[MIT License](LICENSE)
