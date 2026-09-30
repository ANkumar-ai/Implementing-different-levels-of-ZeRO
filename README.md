# ERA V5 · Session 12 — ZeRO 1/2/3 on 32 Virtual GPUs

Notebook: [`zero_stages_32_virtual_gpus.ipynb`](zero_stages_32_virtual_gpus.ipynb) — open in Colab, Run all. Needs only `torch` + `matplotlib`.

## What I built
1. 32 virtual GPUs (ranks 0–31), simulated in one process.
2. Collectives (`reduce_scatter`, `all_gather`) written by hand, so every byte a rank sends is counted.
3. All-reduce is built as reduce-scatter + all-gather, to show the two are the same thing.
4. Toy model: 8 linear layers, width 256, ~526K params. Each layer is one bucket / one FSDP unit.
5. Real mixed-precision state per weight: bf16 weight (2 B) + bf16 grad (2 B) + fp32 master (4 B) + Adam m, v (8 B) = 16 B.
6. Same data, same init → DP, ZeRO-1, ZeRO-2, ZeRO-3.

## Results (N = 32)
Fill from notebook output.

| stage | theory B/param | measured peak B/param | comm / step | optimizer work / rank |
|---|---|---|---|---|
| DP | 16.000 | | | full model |
| ZeRO-1 | 4.375 | | | 1/32 |
| ZeRO-2 | 2.438 | | | 1/32 |
| ZeRO-3 | 0.500 | | | 1/32 |

Checks:
- ZeRO-1/2/3 final weights bit-identical to DP → sharding changes storage, not the math.
- Avg of 32 grads (4 samples each) = grad of one 128-sample batch → DP is a bigger batch, nothing else.

## What I understood

**Why 16 bytes.** bf16 weight and grad for the arithmetic, fp32 master because tiny updates vanish in bf16 rounding, Adam's two fp32 running averages. 30B × 16 B = 447 GiB before any activations.

**DP.** Every GPU holds all 16 B. Each sees different data, all-reduces grads, applies the same update, stays identical. 32 copies of the same state — pure redundancy.

**Collectives.** Reduce-scatter: sum, each rank keeps 1/N. All-gather: each rank's slice → everyone has the full. RS + AG = all-reduce, and in a ring each costs ~P per rank, so all-reduce ≈ 2P.

**ZeRO-1.** Shard the 12 B of optimizer state. Rank r updates only slice r, then all-gathers the new weights. Comm = RS grads + AG weights = still 2P. Free memory.

**ZeRO-2.** Also shard grads. After a layer's grads are reduce-scattered, the full grad is thrown away. Still 2P. My peak is slightly above 2.44 because one layer's full grad exists until its bucket is reduced.

**ZeRO-3.** Shard the weights too. Each layer is all-gathered for forward, dropped, all-gathered again for backward, dropped. That extra gather = 3P. Peak above 0.5 because one gathered layer + its grad sit in memory briefly.

**Compute.** Forward/backward work is identical in every stage. ZeRO removes the redundant optimizer step (each rank does 1/32 of it). The price is only communication, and only for stage 3.

**Why the floor at 4 B.** DP and ZeRO-1 keep full weights + full grads on every card: 4 B × 30B = 111.8 GiB > 74.5 GiB, at any GPU count. So for V5: ZeRO-2 from 32 GPUs or ZeRO-3 from 8.

**Communication cost.** 2P = 120 GB for 30B. NVLink ~0.27 s, InfiniBand ~2.4 s. Faster GPUs shrink compute, not traffic, so the comm fraction rises — which is why overlap (bucketing grads during backward) matters.

## Pros / cons
| stage | pro | con |
|---|---|---|
| DP | simplest, no extra comm | never fits 30B |
| ZeRO-1 | 12 of 16 B sharded for free | weights + grads still replicated |
| ZeRO-2 | ~2 B/param + small, same comm | needs ≥ 32 cards for 30B |
| ZeRO-3 | fits on fewest cards | 1.5× comm, layer-size imbalance across ranks |

## Limits of this simulation
- Ranks run sequentially in one process → wall-clock comm is not real; comm is counted in bytes and converted with NVLink 450 GB/s / IB 50 GB/s.
- Activations not tracked (same across stages).
- No overlap of comm with compute.
