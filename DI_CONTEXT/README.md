# 🎯 DI_CONTEXT - 대학원 컨택 대비 필독 논문
## Context-Aware VLA 연구를 위한 핵심 논문 정리

---

## 📚 논문 리스트 (우선순위 순)

### 🔥 핵심 5편 - Context/Memory 관련

1. **[ELLMER (2024)](./01_ELLMER_2024.md)** ⭐⭐⭐⭐⭐
   - Embodied LLMs with RAG for Complex Robot Tasks
   - Nature Machine Intelligence
   - **왜 중요**: RAG를 VLA에 최초 적용한 선행연구

2. **[RoboMamba (2024)](./02_RoboMamba_2024.md)** ⭐⭐⭐⭐⭐
   - Multimodal State Space Model for Efficient Robot Reasoning
   - NeurIPS 2024
   - **왜 중요**: 메모리 효율적 처리의 새로운 방법

3. **[ReKep (2024)](./03_ReKep_2024.md)** ⭐⭐⭐⭐
   - Spatio-Temporal Reasoning of Relational Keypoint Constraints
   - CoRL 2024 Oral
   - **왜 중요**: 시공간 컨텍스트 처리 방법론

4. **[Episodic Memory (2024)](./04_Episodic_Memory_2024.md)** ⭐⭐⭐⭐
   - Hierarchical Representations of Life-Long Robot Experience
   - arXiv 2024.09
   - **왜 중요**: 선택적 메모리 검색의 실제 구현

5. **[CoT-VLA (2025)](./05_CoT_VLA_2025.md)** ⭐⭐⭐⭐
   - Visual Chain-of-Thought Reasoning for VLA
   - CVPR 2025
   - **왜 중요**: Context를 활용한 복잡한 추론

### 🚀 추가 중요 논문

6. **Helix (2025)** ⭐⭐⭐
   - Dual System Architecture (S1: 200Hz, S2: 7-9Hz)
   - Figure AI
   - **논문 링크**: [https://arxiv.org/abs/2502.02074](https://arxiv.org/abs/2502.02074)

7. **[Pi-Zero (2024)](./Flow_RAG_Papers/Phase1_01_Pi0_2024.md)** ⭐⭐⭐
   - Flow Matching for 50Hz Real-time Control
   - Physical Intelligence
   - **논문 링크**: [https://arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164)

8. **RAM (2024)** ⭐⭐⭐
   - Retrieval-Based Affordance Transfer
   - CoRL 2024 Oral
   - **논문 링크**: [https://arxiv.org/abs/2407.08450](https://arxiv.org/abs/2407.08450)

### 📄 기타 참고 논문들

9. **OpenVLA (2024)**
   - Open-source VLA, 7B parameters
   - **논문 링크**: [https://arxiv.org/abs/2406.09246](https://arxiv.org/abs/2406.09246)
   - **프로젝트**: [https://openvla.github.io/](https://openvla.github.io/)

10. **RT-1 & RT-2 (2023)**
    - Google's Robotics Transformer
    - **RT-1**: [https://arxiv.org/abs/2212.06817](https://arxiv.org/abs/2212.06817)
    - **RT-2**: [https://arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818)

11. **Octo (2024)**
    - Open-source generalist robot policy
    - **논문 링크**: [https://arxiv.org/abs/2405.12213](https://arxiv.org/abs/2405.12213)

---

## 💡 우리 연구의 차별점

### 현재 VLA의 문제점
```python
problems = {
    "ELLMER": "RAG 사용하지만 2Hz로 너무 느림",
    "Pi0": "50Hz로 빠르지만 메모리 없어 반복 실패",
    "OpenVLA": "오픈소스지만 둘 다 해결 못함"
}
```

### 우리의 해결책: Adaptive Context Selection
```python
our_solution = {
    "핵심": "Confidence 기반 선택적 검색",
    "방법": {
        "높은 확신도": "메모리 검색 없이 빠르게",
        "낮은 확신도": "과거 경험 검색하여 정확하게"
    },
    "목표": "20Hz 속도 유지 + 지능적 행동"
}
```

---

## 📖 읽기 전략

### 1단계: 빠른 이해 (각 논문 30분)
- Abstract + Introduction
- 핵심 그림/표
- Conclusion

### 2단계: 깊은 이해 (핵심 3편만 2시간)
- ELLMER: RAG 통합 방법
- RoboMamba: 효율적 메모리 처리
- ReKep: 시공간 추론

### 3단계: 우리 연구와 연결
- 각 논문의 한계점 파악
- 우리 접근법으로 해결 방법 구상
- 교수님께 설명할 스토리 준비

---

## 🎯 컨택 시 활용 방법

### 질문 예시
1. "ELLMER의 2Hz 한계를 어떻게 극복할 수 있을까요?"
2. "RoboMamba의 SSM을 RAG와 결합하면 어떨까요?"
3. "선택적 메모리 검색으로 실시간성을 확보할 수 있을까요?"

### 연구 제안
```
"현재 VLA 모델들이 속도와 지능 사이에서 trade-off를 겪고 있습니다.
저는 Confidence-based Adaptive Retrieval을 통해
필요할 때만 메모리를 검색하는 방식으로
이 문제를 해결하고 싶습니다."
```

---

*마지막 업데이트: 2025년 8월 27일*