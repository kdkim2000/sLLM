# sLLM 과정 학습 정리

이 문서는 "**sLLM 과정**" 전체 학습 내용을 깔끔하게 정리한 최종 요약본입니다.
각 Chapter 별 핵심 요약과 실습 포인트를 담았습니다.

---

## 📚 Chapter별 구성

### Chapter 1 | Introduction

- Transformer 구조 (인코더-디코더)
- GPT, BERT 모델 발전사
- 오픈웨이트(Open-Weight) 모델 vs 클로즈드 모델
- sLLM (Small LLM) 개념 소개
- LLM 최신 트렌드 요약

➡️ [자세히 보기](./chapter1/README.md)


### Chapter 2 | LangChain

- LangChain 프레임워크 소개
- LCEL (LangChain Expression Language) 개념
- Prompt → LLM → Parser 체인 구조 이해
- 문서 검색 챗봇, 외부 API 연결 챗봇 실습

➡️ [자세히 보기](./chapter2/README.md)


### Chapter 3 | RAG (Retrieval Augmented Generation)

- RAG 구조: 검색 → 프롬프트 강화 → 생성
- Semantic, Lexical, Hybrid Search 비교
- MMR 튜닝 전략과 평가 지표 설명
- 다양한 RAG 아키텍처 (RAG-Fusion, Multi-Stage 등)
- 실습: Chroma + LangChain으로 RAG 구축

➡️ [자세히 보기](./chapter3/README.md)


### Chapter 4 | Fine Tuning

- Fine Tuning과 PEFT(LoRA, QLoRA) 이해
- Instruction Tuning, Style Tuning 실습
- 학습률, 배치 크기, Epoch 등 하이퍼파라미터 튜닝 전략
- 실습: LoRA/QLoRA 적용 학습, Instruction 데이터 기반 튜닝

➡️ [자세히 보기](./chapter4/README.md)


### Chapter 5 | Serving / Deployment

- Serving 전략: FastAPI, vLLM 고속 서버 구축
- 모델 최적화(Quantization, Caching)
- Docker로 배포, Kubernetes로 확장 자동화
- Streaming 응답 구조 구현
- Multi-Model Serving 구조 (요청별 모델 라우팅)

➡️ [자세히 보기](./chapter5/README.md)


---

## 🛠️ 실습 프로젝트 예시 (추천)

- FastAPI 기반 LLM API 서버 만들기
- Docker + Kubernetes를 통한 모델 서비스 배포
- Chroma + LangChain 기반 검색 강화 챗봇 구축
- LoRA/QLoRA 기반 맞춤형 Instruction Tuning 모델 개발


## 🏆 최종 목표

- 오픈모델 기반 LLM 실전 프로젝트 독자 수행 능력 확보
- 튜닝-서빙-최적화-배포까지 **엔드투엔드(End-to-End)** 이해 및 실습


---

> 본 문서는 **sLLM 전문 과정**의 총 학습 정리용입니다.
> 추후 버전 업데이트 시, 최신 아키텍처와 사례를 추가할 예정입니다.

---


## 📢 참고 자료

- [변형호 GitHub](https://github.com/NotoriousH2)
- [LangChain 공식 문서](https://python.langchain.com/)
- [Hugging Face Hub](https://huggingface.co/)
- [RAG 관련 논문](https://arxiv.org/abs/2005.11401)
- [LoRA 논문](https://arxiv.org/abs/2106.09685)

---

