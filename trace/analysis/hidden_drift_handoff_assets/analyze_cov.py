#!/usr/bin/env python3
"""improved precision & recall + 클러스터별 커버리지.
커버리지(recall) = 실제 예제 중 '생성물이 도달한' 비율 (실데이터 k-NN 반경 기준)
정밀도(precision) = 생성 예제 중 '실제 분포 안'인 비율"""
import json, sys
from pathlib import Path
import numpy as np

SP = Path("/tmp/claude-1000/-home-seonghyeonnoh-yemokoo/3c6c5124-da9d-47eb-b6b9-ac58b9fffba3/scratchpad")
D = Path("/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace")
K = 3

def dist(a, b):            # 정규화 벡터 → 코사인 거리
    return 1.0 - a @ b.T

def radii(X, k=K):
    d = dist(X, X); np.fill_diagonal(d, np.inf)
    return np.sort(d, axis=1)[:, k - 1]

def pr(real, gen):
    r_real = radii(real)
    d = dist(real, gen)
    covered = (d <= r_real[:, None]).any(axis=1)          # 실제 예제가 덮였나
    r_gen_side = (dist(gen, real) <= r_real[None, :]).any(axis=1)   # 생성이 실제 매니폴드 안인가
    return covered, covered.mean(), r_gen_side.mean()

def kmeans(X, k=10, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        a = np.argmin(dist(X, C), axis=1)
        for j in range(k):
            if (a == j).sum():
                C[j] = X[a == j].mean(0); C[j] /= np.linalg.norm(C[j])
    return np.argmin(dist(X, C), axis=1)

def main():
    print(f"{'task':13}{'set':10}{'커버리지':>9}{'정밀도':>8}   (k=3 매니폴드)")
    worst = {}
    for f in sorted((SP / "cov_emb").glob("*.npz")):
        task = f.stem; z = np.load(f); real = z["real"]
        for tag in ("firstgen", "dawn7"):
            if tag not in z: continue
            covered, cov, prec = pr(real, z[tag])
            print(f"{task:13}{tag:10}{100*cov:8.1f}%{100*prec:7.1f}%")
            if tag == "firstgen":
                lab = kmeans(real)
                rows = [(c, covered[lab == c].mean(), int((lab == c).sum())) for c in range(lab.max() + 1)]
                rows.sort(key=lambda r: r[1])
                worst[task] = (rows, lab)
        print()
    print("=== 첫 생성이 가장 못 덮은 실데이터 클러스터 ===")
    for task, (rows, lab) in worst.items():
        prompts = [r["prompt"] for r in json.load((D / task / "train.json").open())[:1000]]
        print(f"\n[{task}] 클러스터별 커버리지: " + " ".join(f"{100*c:.0f}%({n})" for _, c, n in rows))
        for cid, cov, n in rows[:2]:
            idx = np.where(lab == cid)[0][:2]
            print(f"  - 커버리지 {100*cov:.0f}% (n={n}) 예시:")
            for i in idx:
                t = prompts[i].replace("\n", " ")
                print(f"      {t[:150]}")
main()
