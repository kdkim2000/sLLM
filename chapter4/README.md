# Chapter 4: Fine Tuning

## 📖 개요

Fine Tuning은 기존에 사전 학습된(Pretrained) LLM 모델에 대해,
특정 도메인이나 작업(Task)에 맞는 추가 학습을 수행하는 과정입니다.

이를 통해 모델이 더 정확하고, 상황에 맞는 답변을 하도록 최적화할 수 있습니다.

---

## 4.1 Fine Tuning이 필요한 이유

| 이유 | 설명 |
|:---|:---|
| 도메인 특화 | 의료, 법률, 금융 등 전문 지식 학습 필요 |
| 스타일 변경 | 답변 톤, 길이, 말투 조정 |
| 성능 향상 | 특정 유형 문제에서 더 높은 정확도 달성 |
| 기능 추가 | 신규 명령/질문 양식에 맞춘 답변 지원 |


## 4.2 Fine Tuning 과정 흐름

```mermaid
graph TD
A[Pretrained Model] --> B[Fine Tuning 데이터 준비]
B --> C[모델 학습 (Fine Tuning)]
C --> D[Fine Tuned Model]
```

**요약**:
- 학습 데이터를 준비한다
- 기존 모델을 기반으로 추가 학습을 수행한다
- Fine-Tuned 모델을 저장하고 배포한다


## 4.3 Fine Tuning 데이터 종류

| 데이터 타입 | 설명 | 예시 |
|:---|:---|:---|
| Corpus 데이터 | 일반 텍스트 코퍼스 | 뉴스, 논문, 위키 문서 |
| QA 데이터 | 질문-답변 쌍 | 고객 지원 FAQ |
| Instruction 데이터 | 명령-응답 형식 | "요약해줘", "코드 작성해줘" |
| Style 데이터 | 말투, 길이 등 스타일 변화 데이터 | 비즈니스 톤 ↔ 캐주얼 톤 변환 |


## 4.4 Fine Tuning 주요 방법

| 방법 | 설명 |
|:---|:---|
| Full Fine Tuning | 전체 모델 파라미터 업데이트 (비용, 메모리 소모 큼) |
| PEFT (Parameter Efficient Fine Tuning) | 일부분만 업데이트 (효율적) |

- 대표적인 PEFT 기법: **LoRA (Low-Rank Adaptation)**


## 4.5 LoRA (Low-Rank Adaptation) 이해

```mermaid
graph TD
W[Pretrained Weights (W)] --> BA[Low-rank Matrix A & B 추가]
BA --> Wnew[Modified Weights (W + BA)]
```

**핵심 아이디어**:
- 기존 가중치 W는 고정
- 작은 행렬(A,B)만 학습하여 효율적 업데이트 수행

| 항목 | Full Fine Tuning | LoRA |
|:---|:---|:---|
| 메모리 소모 | 매우 큼 | 매우 작음 |
| 학습 속도 | 느림 | 빠름 |
| Catastrophic Forgetting 위험 | 있음 | 낮음 |


## 4.6 Fine Tuning 실습 예제 (LoRA 적용)

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

# 모델 불러오기
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# LoRA 설정
config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none"
)
model = get_peft_model(model, config)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./lora_ft_model",
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    num_train_epochs=3
)

# Trainer로 학습 진행
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=your_train_dataset
)
trainer.train()
```

> ✅ 실제 데이터셋(`your_train_dataset`)을 준비해 학습해야 합니다.


## 4.7 Fine Tuning 최적화 포인트

| 항목 | 전략 |
|:---|:---|
| 학습률 (learning rate) | 2e-4 ~ 5e-5 범위에서 탐색 |
| 배치 크기 (batch size) | 메모리 허용 범위 내 최대값 추천 |
| Epoch 수 | 3~5회 (작은 데이터셋 기준) |
| Early Stopping | 검증 성능 감소시 학습 중단 |

> ✅ 하이퍼파라미터를 조정하며 Overfitting 방지


## 4.8 Fine Tuning 주의사항

- 데이터 품질이 가장 중요 (잘못된 데이터는 성능 악화)
- 너무 많은 Epoch → Catastrophic Forgetting 위험
- LoRA 사용 시, 적절한 `target_modules` 선택 필수 (q_proj, v_proj 등)
- Fine-Tuned 모델은 원본과 별도로 관리


---

# 📌 요약 키워드

- Fine Tuning
- Instruction Tuning
- LoRA (Low-Rank Adaptation)
- PEFT (Parameter Efficient Fine Tuning)
- Catastrophic Forgetting
- 학습 최적화 전략

---

✅ 표, 다이어그램, 코드 예제, 최적화 가이드까지 포함해 Chapter 4를 체계적으로 정리했습니다.
✅ 추가로 "Instruction Tuning 실습"이나 "Style Tuning 방법"도 이어서 확장 가능합니다!
