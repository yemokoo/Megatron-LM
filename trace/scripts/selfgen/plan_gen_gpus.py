#!/usr/bin/env python3
"""Spread a round's generation over every GPU.

One task per GPU leaves most cards idle whenever a round has few tasks, and the
tail of a round is one long task (MeetingBank) on a single card.  This assigns
each task at least one GPU and hands the spare cards to the expensive tasks
first, so a round finishes in roughly (total cost / 8) instead of (cost of the
slowest task).

usage: plan_gen_gpus.py <n_gpus> <task:cost> [...]   ->  "task<TAB>gpu,gpu,..."
"""
import sys

n_gpus = int(sys.argv[1])
items = []
for spec in sys.argv[2:]:
    task, cost = spec.rsplit(":", 1)
    items.append([task, float(cost), 1])          # every task starts with one GPU

spare = n_gpus - len(items)
while spare > 0:
    # give the next GPU to whichever task currently has the worst cost-per-GPU
    worst = max(items, key=lambda it: it[1] / it[2])
    worst[2] += 1
    spare -= 1

gpu = 0
for task, _cost, shards in items:
    assigned = [str((gpu + k) % n_gpus) for k in range(shards)]
    gpu += shards
    print(f"{task}\t{','.join(assigned)}")
