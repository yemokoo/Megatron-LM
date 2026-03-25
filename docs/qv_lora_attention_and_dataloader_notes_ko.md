# QV LoRA Attention / Data Loader 정리

이 문서는 현재 repo 기준으로 `attention Q/V에 routed LoRA`를 붙인 경로와, `bin/idx` 데이터가 실제 학습 배치로 들어가는 경로를 빠르게 파악하기 위한 메모다.

목표는 다음 두 가지다.

1. 기존 Megatron GPT 경로에서 어디서 갈라져서 우리 커스텀 attention으로 들어가는지 빠르게 찾기
2. `*.bin` / `*.idx` 데이터가 어떤 스크립트와 코드 경로를 거쳐 모델 입력으로 들어가는지 정리하기


## 1. 큰 흐름 요약

현재 wiki QV LoRA pretrain 실행 흐름은 아래와 같다.

1. [pretrain_wiki_qv_lora_local_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/pretrain_wiki_qv_lora_local_bf16.sh)
2. [flame-qv-lora-experts.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/configs/model/flame-qv-lora-experts.sh)
3. `torch.distributed.run ... pretrain_gpt.py --spec megatron.core.models.gpt.qv_lora_layer_specs gpt_qv_lora_local_spec`
4. [pretrain_gpt.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/pretrain_gpt.py) 의 `model_provider()`
5. [qv_lora_layer_specs.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/models/gpt/qv_lora_layer_specs.py)
6. [qv_lora_attention.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/transformer/qv_lora_attention.py)

핵심은 `pretrain_gpt.py`가 `--spec`로 받은 layer spec을 import하고, 그 spec 안에서 기본 `SelfAttention` 대신 `QVLoraSelfAttention`을 사용하도록 바뀐다는 점이다.


## 2. 기존 GPT 경로에서 어디서 갈라지는가

### 2.1 실행 스크립트에서 spec 지정

[flame-qv-lora-experts.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/configs/model/flame-qv-lora-experts.sh#L5)

```bash
--spec megatron.core.models.gpt.qv_lora_layer_specs gpt_qv_lora_local_spec
```

이 한 줄이 기존 GPT layer spec이 아니라 `qv_lora_layer_specs.py`를 쓰게 만드는 출발점이다.


### 2.2 pretrain_gpt.py 에서 spec import

[pretrain_gpt.py:99](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/pretrain_gpt.py#L99)

```python
if args.spec is not None:
    transformer_layer_spec = import_module(args.spec)
```

즉 `args.spec`가 있으면 기본 local/TE GPT layer spec 대신 커스텀 spec을 그대로 사용한다.


### 2.3 qv_lora_layer_specs.py 에서 custom attention 연결

[qv_lora_layer_specs.py:15](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/models/gpt/qv_lora_layer_specs.py#L15)

여기서 `TransformerLayer -> self_attention` 자리에 기본 `SelfAttention` 대신 [QVLoraSelfAttention](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/transformer/qv_lora_attention.py#L115)을 꽂는다.

핵심 부분:

```python
self_attention=ModuleSpec(
    module=QVLoraSelfAttention,
    params={"attn_mask_type": AttnMaskType.causal},
    submodules=SelfAttentionSubmodules(
        linear_qkv=ColumnParallelLinear,
        core_attention=DotProductAttention,
        linear_proj=RowParallelLinear,
        q_layernorm=IdentityOp,
        k_layernorm=IdentityOp,
    ),
)
```

즉 기존 attention의 나머지 구성은 최대한 유지하고, `self_attention` 모듈 자체만 교체하는 구조다.


## 3. QV LoRA Attention은 기존 대비 무엇이 다른가

핵심 구현 파일은 [qv_lora_attention.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/transformer/qv_lora_attention.py) 다.

### 3.1 기존 attention 대비 차이

기존 `SelfAttention`:

1. `hidden_states`에서 `linear_qkv`로 기본 Q/K/V projection 생성
2. attention score 계산
3. softmax
4. V와 곱해 context 생성
5. output projection

현재 `QVLoraSelfAttention`:

1. 부모 `SelfAttention`의 `get_query_key_value_tensors()`를 먼저 호출해 기본 Q/K/V 생성
2. 같은 `hidden_states`를 입력으로 routed LoRA delta를 계산
3. `query += q_delta`, `value += v_delta`
4. 이후 core attention은 기존 경로 그대로 사용

즉 **base attention을 갈아엎은 것이 아니라, Q/V projection 결과에 delta를 더하는 방식**이다.


### 3.2 LoRA가 실제로 어디에 붙는가

[qv_lora_attention.py:163](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/transformer/qv_lora_attention.py#L163)

```python
query, key, value = super().get_query_key_value_tensors(hidden_states, key_value_states)
...
q_delta, v_delta = self.qv_lora_experts(hidden_states)
query = query + q_delta.to(query.dtype).reshape_as(query)
value = value + v_delta.to(value.dtype).reshape_as(value)
```

정리:

- `K`는 현재 수정하지 않음
- `Q`, `V`만 보정
- 보정 위치는 `Q/K/V 기본 projection 이후`, `core attention 이전`


### 3.3 지금 라우팅 의미는 무엇인가

[qv_lora_attention.py:117](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/transformer/qv_lora_attention.py#L117)

```python
router_logits = F.linear(router_input, self.router_weight)
router_probs = torch.softmax(router_logits, dim=-1)
expert_scores, expert_idx = torch.max(router_probs, dim=-1)
```

의미:

- 토큰별로 top-1 expert를 선택
- 각 토큰은 `expert_idx[token]` 하나를 가짐
- `expert_scores[token]`는 선택된 expert의 score


### 3.4 Q와 V는 같은 expert를 쓰는가

쓴다. 단, **expert id는 공유하지만 Q/V 파라미터는 서로 다르다.**

[qv_lora_attention.py:45](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/transformer/qv_lora_attention.py#L45)

- `q_lora_a`, `q_lora_b`
- `v_lora_a`, `v_lora_b`

[qv_lora_attention.py:124](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/transformer/qv_lora_attention.py#L124)

`expert_idx`와 `expert_scores`는 Q/V가 공유한다.

예:

- 어떤 토큰이 `expert 2`로 라우팅되면
- Q는 `q`의 `expert 2` LoRA를 사용
- V는 `v`의 `expert 2` LoRA를 사용

즉 **인덱스는 같고, 파라미터는 다르다.**


### 3.5 현재 grouped 구현이 하는 일

[qv_lora_attention.py:70](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/transformer/qv_lora_attention.py#L70)

현재는 `_compute_grouped_qv_deltas()`를 사용한다.

핵심 흐름:

1. `hidden_states`를 `[num_tokens, hidden]`로 flatten
2. router로 토큰별 `expert_idx`, `expert_scores` 계산
3. expert별로 `token_indices = torch.nonzero(expert_idx == expert_id)` 생성
4. 같은 expert로 간 토큰들만 모아서 Q/V low-rank delta 계산
5. 결과를 원래 토큰 위치에 `index_copy_`로 다시 넣음

이렇게 한 이유는, 예전 구현처럼

```python
lora_a.index_select(0, expert_idx)
lora_b.index_select(0, expert_idx)
```

로 토큰마다 expert weight를 복제하면 메모리 이동과 임시 텐서가 너무 커져서 병목이 심했기 때문이다.


## 4. 우리가 해결한 병목 요약

초기 병목 원인:

1. 토큰별 expert weight 복제
2. Q와 V를 같은 routing 결과로 쓰면서도 거의 별도 경로로 계산
3. LoRA router 쪽에서만 fp32 cast를 강제
4. 이후 non-contiguous tensor가 `view()`를 만나면서 attention 내부 오류 발생

해결 방식:

1. token-wise `index_select` 제거
2. expert별 grouped 처리로 변경
3. Q/V가 같은 routing 결과를 재사용
4. router는 `params_dtype`에 맞춰 bf16 경로로 정리
5. attention 내부 `view()`를 `reshape()`로 교체해 non-contiguous tensor 허용

효과:

- wiki pretrain smoke에서 TFLOP/s가 의미 있게 개선됨
- `mb48`까지 실험 가능한 수준으로 개선됨


## 5. pretrain_gpt.py 에서 우리 repo가 기존 대비 추가한 것

여기서는 “이 repo 기준으로 커스텀 흔적이 있는 부분”을 정리한다.
Megatron upstream과 1:1 diff를 완벽히 비교한 것은 아니므로, 아래는 **현재 repo에서 실험상 중요한 변경/확장 포인트** 중심이다.

### 5.1 `--spec` 기반 custom layer spec 경로

[pretrain_gpt.py:99](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/pretrain_gpt.py#L99)

이 경로 덕분에 `qv_lora_layer_specs.py`를 통해 custom attention으로 분기할 수 있다.

이 부분은 새로운 attention variant를 추가할 때 가장 먼저 보는 entry point다.


### 5.2 teacher KL distillation 경로

[pretrain_gpt.py:282](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/pretrain_gpt.py#L282)

```python
teacher_model = get_old_moe_distill_teacher()
...
student_logits = model(...)
teacher_logits = teacher_model[0](...)
```

그리고 [pretrain_gpt.py:199](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/pretrain_gpt.py#L199) 에서 KL loss를 더한다.

의미:

- continual 단계에서는 student 외에 teacher model도 같이 forward
- teacher는 학습하지 않고 KL target만 제공
- 이 때문에 continual 단계가 wiki pretrain보다 훨씬 느려진다


### 5.3 probe evaluation 경로

[pretrain_gpt.py:376](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/pretrain_gpt.py#L376)

- `_build_probe_dataloader()`
- `_run_single_probe_evaluation()`
- `run_probe_evaluation()`

의미:

- train과 별개로 probe dataset용 dataloader를 따로 만들고
- iteration 주기마다 next-token accuracy / ppl을 측정
- `secondary_probe`까지 별도로 지원

이건 upstream 기본 pretrain 스크립트에서 바로 보이지 않는, 현재 실험 흐름의 중요한 확장 포인트다.


### 5.4 데이터 blend 파싱

[pretrain_gpt.py:321](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/pretrain_gpt.py#L321)

`core_gpt_dataset_config_from_args()`에서 최종 `GPTDatasetConfig`를 만든다.

`--data-path`는 스크립트에서 `1.0 prefix1 1.0 prefix2 ...` 형태로 넘기고, 여기서 `blend`로 해석된다.


## 6. bin/idx 데이터가 모델 입력으로 들어가는 경로

이 부분은 실제 실험 운영에서 매우 중요하다.

### 6.1 실험 스크립트에서 `*.bin`/`*.idx` prefix 문자열 생성

[pretrain_wiki_qv_lora_local_bf16.sh:21](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/pretrain_wiki_qv_lora_local_bf16.sh#L21)

`build_data_path()`는 dataset 디렉토리 안의 모든 `*.bin`을 찾아 prefix로 바꾼다.

예:

- `train_text_document.bin`
- `train_text_document.idx`

이면 `train_text_document` prefix를 만들어 `--data-path`에 넘긴다.


### 6.2 dataset staging

[pretrain_wiki_qv_lora_local_bf16.sh:132](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/pretrain_wiki_qv_lora_local_bf16.sh#L132)

```bash
rsync -rlptD --info=progress2 "$TRAIN_DATASET/" "$SSD_TRAIN_DATASET/"
```

즉 원본 dataset 디렉토리를 `/tmp/flame-moe/<RUN_ID>/dataset/...` 아래로 staging한 뒤, 학습은 이 SSD staging 경로를 사용한다.


### 6.3 `--data-path`로 pretrain_gpt.py 전달

[pretrain_wiki_qv_lora_local_bf16.sh:224](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/pretrain_wiki_qv_lora_local_bf16.sh#L224)

```bash
--data-path $(build_data_path "$SSD_TRAIN_DATASET")
```

Probe도 같은 방식:

- [pretrain_wiki_qv_lora_local_bf16.sh:241](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/pretrain_wiki_qv_lora_local_bf16.sh#L241)
- [pretrain_wiki_qv_lora_local_bf16.sh:248](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/pretrain_wiki_qv_lora_local_bf16.sh#L248)


### 6.4 GPTDataset 생성

[pretrain_gpt.py:347](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/pretrain_gpt.py#L347)

`train_valid_test_datasets_provider()`가:

1. `core_gpt_dataset_config_from_args()`로 config 생성
2. `BlendedMegatronDatasetBuilder(...)`
3. `GPTDataset` / `MockGPTDataset`

를 통해 실제 train/valid/test dataset 객체를 만든다.


### 6.5 dataloader 생성

[training.py:2384](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/training/training.py#L2384)

`build_train_valid_test_data_iterators()` -> [training.py:2326](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/training/training.py#L2326) `build_train_valid_test_data_loaders()` -> [training.py:2356](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/training/training.py#L2356) `build_pretraining_data_loader()`

즉 최종적으로는 legacy sampler 기반 pretraining dataloader를 사용한다.


### 6.6 모델 입력 batch 생성

[pretrain_gpt.py:151](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/pretrain_gpt.py#L151)

`get_batch()`가 dataloader iterator에서 받은 batch를:

- TP rank 기준으로 나누고
- CP rank 기준으로 자른 뒤
- `tokens, labels, loss_mask, attention_mask, position_ids`

형태로 반환한다.

그 다음 [pretrain_gpt.py:275](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/pretrain_gpt.py#L275) 에서 실제 모델 forward로 들어간다.


## 7. 앞으로 어디를 건드리면 되는가

### 7.1 QV LoRA 구조를 바꾸고 싶을 때

가장 먼저 볼 파일:

- [qv_lora_attention.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/transformer/qv_lora_attention.py)

여기서 바꾸는 것:

- 라우팅 방식
- expert별 계산 방식
- Q/V 공유 여부
- K까지 확장할지 여부
- delta 적용 위치


### 7.2 “기본 attention 대신 어떤 모듈을 꽂을지” 바꾸고 싶을 때

- [qv_lora_layer_specs.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/models/gpt/qv_lora_layer_specs.py)

여기서 `self_attention=ModuleSpec(...)` 부분을 바꾸면 된다.


### 7.3 실험 하이퍼파라미터 / 실행 경로를 바꾸고 싶을 때

- [flame-qv-lora-experts.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/configs/model/flame-qv-lora-experts.sh)
- [pretrain_wiki_qv_lora_local_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/pretrain_wiki_qv_lora_local_bf16.sh)
- [continual_code_from_wiki_qv_lora_expand_local_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/continual_code_from_wiki_qv_lora_expand_local_bf16.sh)


### 7.4 probe / KL teacher / extra evaluation을 바꾸고 싶을 때

- [pretrain_gpt.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/pretrain_gpt.py)

특히:

- `forward_step()`
- `loss_func()`
- `_build_probe_dataloader()`
- `_run_single_probe_evaluation()`
- `run_probe_evaluation()`


## 8. 짧은 결론

현재 repo의 QV LoRA는:

- 기본 GPT layer spec 대신 custom spec으로 분기하고
- `QVLoraSelfAttention`이 Q/V projection 결과에 routed LoRA delta를 더한 뒤
- 나머지 core attention은 기존 경로를 그대로 사용한다.

데이터는:

- 실험 스크립트가 `bin/idx` dataset을 SSD staging으로 복사하고
- prefix 형태의 `--data-path`를 만들고
- `GPTDataset -> pretraining dataloader -> get_batch()` 순으로 모델 입력으로 들어간다.

이 문서를 기준으로 보면, 이후 새로운 기능을 붙일 때 우선적으로 건드릴 곳은 다음 4군데다.

- attention 로직: [qv_lora_attention.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/transformer/qv_lora_attention.py)
- spec 분기점: [qv_lora_layer_specs.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/megatron/core/models/gpt/qv_lora_layer_specs.py)
- 실행/하이퍼파라미터: [pretrain_wiki_qv_lora_local_bf16.sh](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/scripts/experiment/a100/pretrain_wiki_qv_lora_local_bf16.sh)
- probe / KL / batch 경로: [pretrain_gpt.py](/Users/yemokoo/miil/1.%20LLM-CL/LLM-continual-learning/Megatron-LM/pretrain_gpt.py)
