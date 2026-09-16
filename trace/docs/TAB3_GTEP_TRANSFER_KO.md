# Table 3 — GTEP 전이 실험 설계 (준비 완료, 실행 전)

"Hyperparameters in Continual Learning: A Reality Check" (arXiv:2403.09066)의
2단계 프로토콜을 따른다. Phase 1(wiki→code→conversation, G2)에서 튜닝한 HP를
Phase 2(TRACE 8-task)에 **검증 없이 그대로** 전이해서, "conventional protocol"이
얼마나 과대평가하는지를 보는 것이 목적이다. 그래서 스케일 불일치(예: EWC Fisher
정규화 방식 차이) 같은 것도 굳이 보정하지 않고 그대로 넣는다 — 그 불일치 자체가
관찰 대상이지, 막을 대상이 아니다.

Table 1(TRACE-native, 자체 튜닝/논문 기본값)과 값이 갈리는 지점은 각주로
표시한다.

## 학습 설정 — Table 1과 동일 (2026-09-13 결정)

lr·스케줄러·warmup·weight decay·seed·배치 등 **공통 학습 설정은 옮기지 않는다.**
TRACE Table 1 계약(lr 2e-4, cosine, warmup 0.03, wd 0, seed 2025, global batch 64,
r64/α128/dropout 0.05)을 그대로 쓰고, **각 방법의 고유 파라미터만** wiki에서 정한
값으로 바꾼다. wiki↔TRACE 설정 diff(`scratchpad/cfgdiff/`)는 참고용으로만 봤고,
review 스케줄·Fisher 추정량·어댑터 dropout 같은 차이도 옮기지 않는다.

## 방법별 고유 파라미터 (이것만 바뀜)

| 방법 | 항목 | 값 | 출처 / 상태 |
|---|---|---|---|
| EWC | λ | **1.8e6** | wiki 5점 스윕(1.2e4/2e5/1.8e6/6e6/2e7), AA 최적(6e6과 사실상 동률, 더 작은 쪽 채택). Table 1의 λ=400(TRACE 기본값)과 스케일 자체가 다름 — 이 불일치를 그대로 노출하는 게 Table 3의 목적 |
| O-LoRA | λ1 (직교) | 0.5 | 공식 O-LoRA `long.sh` 값 |
| O-LoRA | λ2 (L2) | 0 | wiki에서 0.5 vs 0 비교, 0이 AA/code 둘 다 우세해서 채택. **Table 1 O-LoRA run(λ1=0.5, λ2=0)과 값이 같아 그 run을 Table 3 행으로 재사용, 새로 돌리지 않음** |
| SLoRA | rank | 512 | wiki에서 유일하게 스윕한 용량 축 |
| SLoRA | denoising | max | max/min 비교, min은 학습 발산(터짐). Table 1 SLoRA-Pre도 이미 max라 변경 없음 |
| Lifelong-MoE | λ_KL | **1.5** | wiki 6점 스윕(0.05/0.1/0.2/0.5/1.0/1.5). AA는 1.0(.3728)이 1.5(.3713)보다 근소 우위(차이 0.0015, 오차 범위)지만 FM은 1.5(+.163)가 1.0(+.183)보다 뚜렷이 좋음(차이 0.02) → **AA 근소 차이보다 FM 개선을 우선해 1.5로 확정**(2026-09-13 재검토, 최초 채택은 1.0이었음) |
| MoE-LPR | γ | **0.01** | 최종 근거는 재스윕 `moelpr_gamma_2gpu_nongrouped_20260911/`(γ=0.01/0.05/0.1/0.5/1/5). 정의는 `collect_ours_hp.py`와 동일: a11=wiki 소스 0.460047(`wiki_source_probe_20260908/ffn.probe`, 같은 e8 체크포인트), a22=code 단계(router-lpr) 끝 code, 최종=conversation router-lpr 끝. 결과(LA/FM/AA): 0.01 .4999/.0023/.4984 · 0.05 .5000/.0024/.4984 · 0.1 .4996/.0024/.4980 · 0.5 .4979/.0044/.4950 · 1 .4962/.0062/.4921 · 5 .4810/.0054/.4775. 항등식 잔차 ~0. γ=0.01이 FM 최저·AA 공동 최고라 채택. 구버전 `hp_ablation_20260906` 스윕의 γ=0.5 유망값은 world_size=4 셀로, 1-GPU 재현(`moelpr-g0.5-w1/`) 실패해 폐기 |

## 실행 구성

큐: `/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/tab3_20260913/queue.sh`
(`DRY=1 bash queue.sh`로 명령 확인). 각 job은 빈 GPU 페어에서 학습 후 같은 페어로
sparse-15 평가까지 한다. Table 3와 O-LoRA 스윕을 섞어서 배치한다.

| run | 바뀌는 값 | micro batch (global 64) |
|---|---|---|
| tab3_ewc_l1.8e6 | EWC λ=1.8e6 | 16 (예전 mb16 OOM은 Fisher 단계였고 배치 1로 수정됨) |
| tab3_slora_r512 | SLoRA lora_r=512 (`train_trace.sh`의 `SLORA_LORA_R`, 기본 64) | 8 |
| tab3_lifelong_kd1.5 | Lifelong KD λ=1.5 | 8 |
| tab3_moelpr_g0.01 | MoE-LPR γ=0.01 | 16 |
| (Table 1 O-LoRA 재사용) | λ1=0.5, λ2=0 | — |

**O-LoRA λ 스윕 (Table 1 채택값 결정용, Table 1 계약 그대로)**: λ1∈{0.05, 0.5, 5} ×
λ2∈{0, 0.1}. λ1=0.5/λ2=0은 기존 Table 1 run이라 새로 도는 건 5칸
(`olora_l1-0.05_l2-0`, `olora_l1-0.5_l2-0.1`, `olora_l1-5_l2-0`, `olora_l1-0.05_l2-0.1`,
`olora_l1-5_l2-0.1`). AA/FM 기준으로 가장 나은 칸이 Table 1 O-LoRA 행이 된다.

## 남은 것

- 실행 시작(사용자 지시 대기). 지금 Table 1 평가 4개가 GPU 8장을 쓰는 중이라 큐는
  빈 페어가 생기는 대로 들어간다.
- SLoRA r512는 메모리 확인 전이라 micro batch 8로 잡음. 첫 태스크 로그에서 OOM 여부 확인.
