# Comparing workflow admission policies

A workflow has a root ID and one or more agents. The replay uses causal
dependencies between model calls and represents subagent forks and joins.
The scheduler allocates a reference share of capacity to each workflow with
outstanding inference, split among its outstanding agents. Tool-only gaps earn no
entitlement. The fairness score divides accumulated entitlement by completed
standalone-equivalent work. Before the first completion, the scheduler uses the
smallest estimated cost among the workflow’s outstanding turns as the denominator.

| Policy | Admission at a shared concurrency limit |
| --- | --- |
| Bypass | Admit into a free slot or return a capacity rejection. |
| FIFO | Queue excess work and admit by arrival order when a slot opens. |
| Fair | Use the same queue bounds and admit from the workflow furthest behind its share. |

The router uses one scheduler and one engine target. It changes admission order
when a slot opens. Engine batching, preemption, KV placement, and GPU capacity
remain under the engine’s control. The scheduler uses per-turn costs rather than
the per-iteration estimates described in the [Duru paper, §5](https://arxiv.org/abs/2611.23558).
Replay clients send workflow, agent, and cost information in experiment headers,
which the router consumes before forwarding the request.

## Compare three policies on a private GPU

Use an engine that supports streaming Chat Completions, usage reports, fixed output
lengths via `max_tokens` with `ignore_eos`, and the vLLM `cache_salt` parameter.
Run the router, engine, and driver on the same machine. Record your engine image,
weights, flags and resource limits in an `engine.json` file. The driver checks the
manifest digest across arms. Inspect and verify the actual engine configuration
separately.

1. Start the router in bypass mode. Set the upstream, served model, and desired
   capacity, and carry the same settings through the scored arms.

   ```sh
   .venv/bin/python router.py --upstream http://127.0.0.1:8001 --port 8088 \
     --served-model local-model --capacity 8 --queue-timeout 180 \
     --request-timeout 300 --mode bypass
   ```

2. From another terminal, measure each generated turn in isolation. The driver
   includes each turn’s causal prompt history and reuses these costs across arms.

   ```sh
   .venv/bin/python gpu_replay.py calibrate \
     --url http://127.0.0.1:8088 --engine-url http://127.0.0.1:8001 \
     --workloads workloads.json --scenario mixed --seed 42 \
     --engine-manifest engine.json --capacity 8 --request-timeout 300 \
     --out results/calibration
   ```

3. Score bypass first. For FIFO and fair, restart only the local router using the
   corresponding mode, keeping every other setting fixed. Set the shell variable
   `mode` to match the running router for each invocation.

   ```sh
   mode=bypass
   .venv/bin/python gpu_replay.py run-arm \
     --url http://127.0.0.1:8088 --engine-url http://127.0.0.1:8001 \
     --workloads workloads.json --scenario mixed --seed 42 \
     --engine-manifest engine.json --calibration results/calibration \
     --capacity 8 --request-timeout 300 --root-timeout 600 --time-scale 0.01 \
     --mode "$mode" --out "results/$mode"
   ```

4. Verify the paired inputs and compare outputs.

   ```sh
   .venv/bin/python gpu_replay.py compare --out results/comparison.json \
     results/bypass results/fifo results/fair
   ```

Each scored arm writes request and workflow CSVs, a JSON summary, and engine
metrics captured before and after the arm. Fresh salts separate prefix reuse
across roots and arms while retaining reuse within each root. Cache contents from
previous arms can still occupy memory. Repeat arm orderings for consequential
comparisons.

## Read the results

- Count outcomes across all assigned workflows first. Rejections and failures
  change the mix of work included in successful percentiles.
- Compare FIFO and fairness on shared completed workflows for normalized
  slowdown. When successful request sets differ, use the same successful request
  IDs from both CSVs for queue wait and time to first token. The driver checks
  that successful responses match their fixed output-token targets.
- Check router queue wait and engine wait independently. A better workflow
  fairness score can coexist with higher request latency.
- A small time scale compresses arrivals and tool gaps to raise offered load.
  Measure realistic arrival rates separately before making serving decisions.
