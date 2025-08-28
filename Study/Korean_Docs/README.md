# 🇰🇷 VLA 연구 한글 가이드

Vision-Language-Action (VLA) 모델 연구를 위한 한글 학습 자료입니다.

## 📚 목차

### 1. AI 기초 개념 (한글 설명)
- [신경망 기초 이해하기](./AI_Fundamentals/01_neural_networks_korean.md)
- [어텐션 메커니즘 완벽 가이드](./AI_Fundamentals/02_attention_korean.md)
- [트랜스포머 구조 상세 설명](./AI_Fundamentals/03_transformer_korean.md)
- [멀티모달 학습 이해하기](./AI_Fundamentals/04_multimodal_korean.md)
- [플로우 모델 설명](./AI_Fundamentals/11_flow_models_korean.md)

### 2. 핵심 논문 한글 요약
- [RT-1: 로보틱스 트랜스포머 첫 번째 모델](./Foundation_Papers/RT-1_korean.md)
- [RT-2: 웹 지식을 활용한 VLA](./Foundation_Papers/RT-2_korean.md)
- [OpenVLA: 오픈소스 VLA 프로젝트](./Foundation_Papers/OpenVLA_korean.md)
- [CLIP: 비전-언어 정렬 모델](./Foundation_Papers/CLIP_korean.md)

### 3. 최신 연구 동향 (2024-2025)
- [π₀(파이제로): 최신 플로우 기반 VLA](./Latest_Trends/Pi0_korean.md)
- [ATM: 임의 지점 궤적 모델링](./Latest_Trends/ATM_korean.md)
- [2025년 VLA 상용화 동향](./Latest_Trends/Commercial_VLA_korean.md)

### 4. RAG 시스템 이해
- [RAG 기본 개념과 원리](./RAG_Systems/RAG_basics_korean.md)
- [Bridge RAG 최적화 기법](./RAG_Systems/Bridge_RAG_korean.md)
- [VLA를 위한 RAG 시스템 설계](./RAG_Systems/RAG_for_VLA_korean.md)

## 🎯 학습 순서 추천

### 초보자 코스 (2주)
1. 신경망 기초 → 어텐션 메커니즘 → 트랜스포머
2. CLIP 논문 → RT-1 논문 → OpenVLA
3. RAG 기본 개념 → VLA를 위한 RAG

### 중급자 코스 (1주)
1. RT-2 → π₀ 모델 → ATM
2. Bridge RAG → 최신 연구 동향
3. 구현 실습 프로젝트

### 고급자 코스 (3일)
1. π₀ 플로우 모델 심화
2. RAG-VLA 통합 아키텍처
3. 실제 로봇 응용

## 💡 핵심 개념 빠른 참조

### VLA란?
**Vision-Language-Action 모델**: 시각 정보(카메라 이미지)와 언어 명령(자연어)을 이해하여 로봇 행동(액션)을 생성하는 AI 모델

### 왜 중요한가?
- 🤖 범용 로봇 제어: 하나의 모델로 다양한 작업 수행
- 🧠 인간 수준 이해: 자연어 명령을 이해하고 실행
- 📈 확장성: 더 많은 데이터로 계속 성능 향상

### 핵심 구성요소
1. **비전 인코더**: 이미지를 이해
2. **언어 인코더**: 명령어를 이해
3. **액션 디코더**: 로봇 동작 생성
4. **트랜스포머**: 정보 통합 및 처리

## 📖 용어 사전

| 영어 | 한글 | 설명 |
|------|------|------|
| VLA | 비전-언어-액션 모델 | 시각과 언어를 이해하여 행동하는 AI |
| Transformer | 트랜스포머 | 어텐션 기반 신경망 구조 |
| RAG | 검색 증강 생성 | 외부 지식을 검색하여 활용하는 기법 |
| Flow Matching | 플로우 매칭 | 연속적인 동작 생성 기법 |
| Zero-shot | 제로샷 | 학습하지 않은 새로운 작업 수행 |
| Imitation Learning | 모방 학습 | 인간 시연을 보고 학습 |
| Embodiment | 임바디먼트 | 로봇의 물리적 형태 |

## 🚀 시작하기

1. **기초 개념 이해**: AI_Fundamentals 폴더의 문서들부터 시작
2. **논문 읽기**: Foundation_Papers의 한글 요약 확인
3. **최신 동향 파악**: Latest_Trends의 2024-2025 연구 확인
4. **실습하기**: 제공된 코드 예제로 직접 구현

## 📬 문의 및 기여

- 오류나 개선사항 발견 시 이슈 등록 환영
- 추가 번역이나 설명 기여 환영

---

*마지막 업데이트: 2025년 8월 26일*
*VLA 연구를 위한 한글 학습 자료*