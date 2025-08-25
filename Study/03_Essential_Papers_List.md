# 📚 VLA 필수 논문 리스트
## 체계적 VLA 연구를 위한 논문 읽기 순서

---

## 🎯 논문 읽기 전략

### 읽기 우선순위
```python
reading_priority = {
    "🔥 Must Read": "반드시 읽어야 할 핵심 논문 (완전 이해 필요)",
    "📖 Should Read": "읽어두면 좋은 중요 논문 (개념 파악 수준)",
    "📚 Could Read": "시간 있을 때 읽는 참고 논문 (훑어보기)",
    "🆕 Latest": "2024-2025 최신 논문 (트렌드 파악용)"
}
```

### 난이도별 분류
- 🟢 **초급**: 기초 개념, 쉬운 설명
- 🟡 **중급**: 전문 지식 필요, 수식 포함  
- 🔴 **고급**: 복잡한 이론, 높은 배경지식 요구

---

## 📖 Week 0-1: VLA 기초 이해

### 🔥 Must Read

#### 1. **RT-1: Robotics Transformer** (2022)
- **Authors**: Google Research
- **Link**: https://arxiv.org/abs/2212.06817
- **난이도**: 🟢 초급
- **읽는 이유**: VLA의 개념을 처음 정립한 논문
- **핵심 아이디어**: Vision + Language → Action의 기본 패러다임
- **Pass 3 필요**: ✅ 완전 이해 필수

#### 2. **OpenVLA: An Open-Source Vision-Language-Action Model** (2024)
- **Authors**: Stanford, UC Berkeley
- **Link**: https://openvla.github.io/
- **난이도**: 🟡 중급
- **읽는 이유**: 현재 SOTA 오픈소스 VLA 모델
- **핵심 아이디어**: 대규모 데이터 + 효율적 아키텍처
- **Pass 3 필요**: ✅ 코드까지 완전 분석

#### 3. **RT-X: Real-World Robotics Data** (2023)
- **Authors**: Google DeepMind
- **Link**: https://arxiv.org/abs/2310.08864
- **난이도**: 🟢 초급
- **읽는 이유**: VLA 학습에 필요한 데이터셋 이해
- **핵심 아이디어**: 다양한 로봇 데이터 통합의 중요성
- **Pass 3 필요**: ✅ 데이터 구조 완전 이해

### 📖 Should Read

#### 4. **RT-2: Vision-Language-Action Models** (2023)  
- **Authors**: Google DeepMind
- **Link**: https://arxiv.org/abs/2307.15818
- **난이도**: 🟡 중급
- **읽는 이유**: RT-1의 발전된 형태, 스케일링 법칙
- **핵심 아이디어**: 웹 데이터 + 로봇 데이터 co-training

#### 5. **PaLM-E: Embodied Multimodal Language Model** (2023)
- **Authors**: Google
- **Link**: https://arxiv.org/abs/2303.03378  
- **난이도**: 🟡 중급
- **읽는 이유**: 멀티모달 + Embodied AI의 결합
- **핵심 아이디어**: 언어 모델에 센서 정보 통합

---

## 🔍 Week 2: RAG 시스템 이해

### 🔥 Must Read

#### 6. **RAG: Retrieval-Augmented Generation** (2020)
- **Authors**: Facebook AI
- **Link**: https://arxiv.org/abs/2005.11401
- **난이도**: 🟡 중급  
- **읽는 이유**: RAG의 원조 논문, 기본 원리 이해
- **핵심 아이디어**: Retrieval + Generation의 결합

#### 7. **ELLMER: Retrieval-Augmented VLA** (2025) ⭐
- **Authors**: [Latest Research]
- **Link**: [To be updated]
- **난이도**: 🔴 고급
- **읽는 이유**: VLA에 RAG를 적용한 첫 시도
- **핵심 아이디어**: 로봇 태스크에서 외부 지식 활용
- **Pass 3 필요**: ✅ 우리 연구의 직접적 관련성

### 📖 Should Read

#### 8. **REALM: Retrieval-Augmented Language Model** (2020)
- **Authors**: Google Research
- **Link**: https://arxiv.org/abs/2002.08909
- **난이도**: 🟡 중급
- **읽는 이유**: Neural retrieval의 선구적 연구

#### 9. **FiD: Fusion-in-Decoder** (2021)
- **Authors**: Facebook AI  
- **Link**: https://arxiv.org/abs/2007.01282
- **난이도**: 🟡 중급
- **읽는 이유**: 검색된 문서들을 효과적으로 융합하는 방법

#### 10. **LangChain: Building applications with LLMs** (2023)
- **Link**: https://github.com/langchain-ai/langchain
- **난이도**: 🟢 초급
- **읽는 이유**: RAG 구현을 위한 실용적 도구
- **핵심**: Documentation + Tutorials

---

## 🤖 Week 3: 로보틱스 기초

### 📖 Should Read

#### 11. **Imitation Learning Survey** (2018)
- **Authors**: Berkeley
- **Link**: https://arxiv.org/abs/1811.02553
- **난이도**: 🟡 중급
- **읽는 이유**: VLA의 기본이 되는 모방 학습 이해

#### 12. **Behavioral Cloning: What and How to Imitate** (2019)
- **Link**: https://arxiv.org/abs/1906.02544  
- **난이도**: 🟢 초급
- **읽는 이유**: BC의 기본 원리와 한계점

### 📚 Could Read

#### 13. **Robot Learning Survey** (2023)
- **Authors**: CMU, MIT
- **난이도**: 🟡 중급
- **읽는 이유**: 전체적인 로봇 학습 분야 조망

#### 14. **Sim-to-Real Transfer Survey** (2023)
- **난이도**: 🟡 중급
- **읽는 이유**: 시뮬레이션 실험의 실제 적용 가능성

---

## 🆕 Week 4: 최신 VLA 동향 (2024-2025)

### 🔥 Must Read

#### 15. **VLA-RL: Online Reinforcement Learning for VLA** (2025) ⭐
- **Authors**: [Latest Research]
- **난이도**: 🔴 고급
- **읽는 이유**: 온라인 학습으로 VLA 개선하는 최신 연구
- **핵심 아이디어**: 실패에서 학습하는 VLA
- **Pass 3 필요**: ✅ SIREN-VLA 아이디어와 직접 연관

#### 16. **SC-VLA: Self-Correcting Vision-Language-Action** (2024)
- **Authors**: [Latest Research]
- **난이도**: 🔴 고급
- **읽는 이유**: 실패 감지 및 수정 메커니즘
- **핵심 아이디어**: 로봇이 스스로 실수를 고치는 방법

### 📖 Should Read  

#### 17. **AHA Model: Failure Analysis in Robotics** (2024)
- **난이도**: 🟡 중급
- **읽는 이유**: 로봇 실패의 원인 분석 및 학습
- **핵심 아이디어**: 실패 패턴의 체계적 분석

#### 18. **PaLI-X: Scaling Vision-Language Models** (2024)
- **Authors**: Google
- **난이도**: 🟡 중급  
- **읽는 이유**: 대규모 VLA 모델의 최신 동향

---

## 🧠 Week 5: Context & Memory 심화

### 🔥 Must Read

#### 19. **Transformer-XL: Attentive Language Models** (2019)
- **Link**: https://arxiv.org/abs/1901.02860
- **난이도**: 🟡 중급
- **읽는 이유**: 긴 컨텍스트 처리 방법의 기초
- **핵심 아이디어**: Segment-level recurrence + relative positional encoding

#### 20. **Longformer: Long-Document Attention** (2020)
- **Link**: https://arxiv.org/abs/2004.05150
- **난이도**: 🟡 중급
- **읽는 이유**: 효율적인 긴 시퀀스 처리

### 📖 Should Read

#### 21. **Neural Episodic Control** (2017)
- **Link**: https://arxiv.org/abs/1703.01988
- **난이도**: 🟡 중급
- **읽는 이유**: 외부 메모리를 활용한 학습

#### 22. **Memory Networks Survey** (2021)
- **난이도**: 🟡 중급
- **읽는 이유**: 다양한 메모리 아키텍처 비교

---

## 🔬 Week 6: Advanced Topics

### 📖 Should Read

#### 23. **Hierarchical Reinforcement Learning Survey** (2019)
- **난이도**: 🟡 중급
- **읽는 이유**: 계층적 태스크 분해 및 관리

#### 24. **Continual Learning in Robotics** (2023)
- **난이도**: 🔴 고급
- **읽는 이유**: 지속적 학습 및 망각 방지

### 📚 Could Read

#### 25. **Meta-Learning Survey** (2022)
- **난이도**: 🔴 고급
- **읽는 이유**: 빠른 적응 능력 향상

#### 26. **Causal Inference in ML** (2021)  
- **난이도**: 🔴 고급
- **읽는 이유**: 실패 원인의 인과관계 분석

---

## 🔮 Week 7-8: Neurosymbolic & Self-Improvement

### 🔥 Must Read (SIREN-VLA 관련)

#### 27. **Neurosymbolic AI Survey** (2024)
- **난이도**: 🔴 고급
- **읽는 이유**: Neural + Symbolic 결합의 최신 동향
- **핵심**: 해석 가능한 AI 시스템 구축

#### 28. **Logic + Neural Networks** (2023)
- **난이도**: 🔴 고급
- **읽는 이유**: 논리적 추론과 신경망의 통합

### 📖 Should Read

#### 29. **Symbolic Reasoning for Robots** (2024)
- **난이도**: 🟡 중급
- **읽는 이유**: 로봇 계획에서 symbolic reasoning 활용

#### 30. **Dual-Process Theory in AI** (2023)
- **난이도**: 🔴 고급
- **읽는 이유**: System 1 (fast) vs System 2 (slow) 사고

---

## 📋 논문별 체크리스트

### 필수 논문 진도 체크 (Must Read)
```markdown
## VLA 기초
- [ ] RT-1 (2022)
- [ ] OpenVLA (2024) 
- [ ] RT-X (2023)

## RAG 시스템  
- [ ] RAG (2020)
- [ ] ELLMER (2025)

## 최신 동향
- [ ] VLA-RL (2025)
- [ ] SC-VLA (2024)

## Context Management
- [ ] Transformer-XL (2019)

## Neurosymbolic (선택)
- [ ] Neurosymbolic AI Survey (2024)
```

### 읽기 상태 표시
```python
paper_status = {
    "📋 Queue": "읽을 예정",
    "📖 Reading": "현재 읽는 중", 
    "✅ Done": "완독 완료",
    "🔄 Review": "재독 필요",
    "⭐ Favorite": "즐겨찾기",
    "💡 Idea": "연구 아이디어 도출"
}
```

---

## 🔍 논문 찾기 및 관리

### 논문 검색 사이트
```python
search_engines = {
    "arXiv.org": "최신 preprint 논문",
    "Google Scholar": "인용수 기반 검색",
    "Semantic Scholar": "관련 논문 추천",
    "Papers with Code": "코드와 함께 제공",
    "Connected Papers": "논문 간 관계 시각화"
}
```

### 논문 관리 도구
```python
management_tools = {
    "Zotero": {
        "장점": "무료, 강력한 기능",
        "용도": "PDF 저장, 주석, 인용 관리"
    },
    
    "Mendeley": {
        "장점": "소셜 기능, 온라인 동기화", 
        "용도": "협업 연구, 논문 공유"
    },
    
    "Obsidian": {
        "장점": "논문 간 연결, 그래프 뷰",
        "용도": "개념 정리, 아이디어 연결"
    }
}
```

### 폴더 구조 제안
```
📁 VLA_Papers/
├── 📁 01_Foundations/
│   ├── RT-1_2022.pdf
│   ├── OpenVLA_2024.pdf
│   └── RT-X_2023.pdf
├── 📁 02_RAG_Systems/
│   ├── RAG_2020.pdf
│   ├── ELLMER_2025.pdf
│   └── REALM_2020.pdf
├── 📁 03_Latest_2024_2025/
│   ├── VLA-RL_2025.pdf
│   └── SC-VLA_2024.pdf
├── 📁 04_Context_Memory/
└── 📁 05_Neurosymbolic/
```

---

## 📊 읽기 진도 추적

### 주차별 목표
```python
weekly_targets = {
    "Week_0": "3편 (RT-1, OpenVLA, RT-X)",
    "Week_1": "5편 (RT-2, PaLM-E + 보완)",
    "Week_2": "4편 (RAG fundamentals)", 
    "Week_3": "3편 (Robotics basics)",
    "Week_4": "4편 (Latest trends)",
    "Week_5": "3편 (Context/Memory)",
    "Week_6": "2편 (Advanced topics)",
    "Week_7": "2편 (Neurosymbolic)",
    
    "총_목표": "30편 (8주간)"
}
```

### 읽기 품질 지표
```python
quality_metrics = {
    "이해도": "논문 내용을 다른 사람에게 설명 가능",
    "연결성": "다른 논문과의 관계 파악",
    "응용성": "내 연구에 어떻게 활용할지 구상", 
    "비판성": "논문의 장단점 분석 가능"
}
```

---

## 💡 효율적 읽기 팁

### 논문별 맞춤 전략
```python
reading_strategies = {
    "기초논문": {
        "방법": "천천히, 완벽 이해",
        "시간": "2-3시간",
        "목표": "개념의 완전한 체화"
    },
    
    "최신논문": {  
        "방법": "빠르게, 핵심만",
        "시간": "1시간",
        "목표": "트렌드 파악, 아이디어 수집"
    },
    
    "관련논문": {
        "방법": "선택적, 필요 부분만",
        "시간": "30분-1시간", 
        "목표": "특정 기법이나 결과 확인"
    }
}
```

### 논문 간 연결고리 만들기
```python
connection_methods = [
    "🔗 공통 키워드로 그룹핑",
    "📈 시간순 발전 과정 추적",  
    "⚡ 방법론별 분류 및 비교",
    "🎯 해결하는 문제별 정리",
    "👥 연구팀/저자별 연구 라인 파악"
]
```

---

## 🎯 다음 단계

논문 리스트를 파악했다면:

1. **즉시 시작**: RT-1부터 읽기 시작
2. **도구 준비**: Zotero 설치 및 설정  
3. **계획 수립**: 8주 읽기 스케줄 작성
4. **템플릿 활용**: `04_Study_Templates/paper_summary_template.md` 사용
5. **함께 읽기**: 어려운 논문은 Claude와 함께!

**첫 논문 선택하셨다면 "같이 읽어요!"라고 말씀해주세요!** 🚀

---

*Created: 2025-08-24*  
*Last Updated: 2025-08-24*  
*Total Papers: 30+ (Priority-based)*  
*Reading Period: 8 weeks*