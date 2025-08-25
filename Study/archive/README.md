# 📚 VLA 연구 논문 아카이브
## 체계적인 논문 수집 및 분류

---

## 📁 폴더 구조

```
archive/
├── README.md                    # 이 파일
├── 01_Foundation_Papers/        # VLA 기초 필수 논문들
├── 02_RAG_Systems/             # RAG 관련 핵심 논문들  
├── 03_Latest_Trends_2024_2025/ # 최신 VLA 연구 동향
├── 04_Context_Memory/          # Context 관리 관련 논문
├── 05_Robotics_Fundamentals/   # 로보틱스 기초 이론
├── 06_Neurosymbolic_AI/        # 신경-상징 AI (SIREN-VLA용)
├── 07_Benchmarks_Datasets/     # 평가 지표 및 데이터셋
└── 08_Implementation_Guides/    # 구현 관련 기술 논문
```

---

## 🎯 논문 분류 기준

### 📊 우선순위
- **🔥 Critical**: 반드시 읽어야 할 핵심 논문
- **📖 Important**: 읽어두면 매우 도움이 되는 논문  
- **📚 Reference**: 필요할 때 참고하는 보조 논문

### 📈 난이도
- **🟢 Beginner**: 기초 개념, 입문용
- **🟡 Intermediate**: 전문 지식 필요
- **🔴 Advanced**: 고도의 배경 지식 요구

### 📅 시기
- **📰 Latest (2024-2025)**: 최신 연구 동향
- **📖 Classic (2020-2023)**: 정립된 기본 이론
- **📜 Historical**: 역사적으로 중요한 논문

---

## 🚀 빠른 시작 가이드

### Week 0-1: 기초 다지기
**추천 읽기 순서**:
1. `01_Foundation_Papers/RT-1_Robotics_Transformer.md`
2. `01_Foundation_Papers/OpenVLA_Paper_Analysis.md`
3. `01_Foundation_Papers/RT-X_Dataset_Overview.md`

### Week 2: RAG 시스템
**추천 읽기 순서**:
1. `02_RAG_Systems/RAG_Original_Paper.md`
2. `02_RAG_Systems/ELLMER_VLA_RAG.md`
3. `02_RAG_Systems/LangChain_Framework.md`

### Week 3-4: 최신 동향
**추천 읽기 순서**:
1. `03_Latest_Trends_2024_2025/VLA_RL_Online_Learning.md`
2. `03_Latest_Trends_2024_2025/SC_VLA_Self_Correction.md`
3. `03_Latest_Trends_2024_2025/AHA_Model_Failure_Analysis.md`

---

## 📋 논문 추가 가이드

### 새 논문 추가할 때
1. **적절한 폴더 선택**
2. **파일명 규칙**: `[Paper_Name]_[Year].md`
3. **템플릿 사용**: `../Study/04_Study_Templates/paper_summary_template.md`
4. **태그 추가**: 우선순위, 난이도, 분류

### 파일명 예시
```
RT-1_Robotics_Transformer_2022.md
OpenVLA_Open_Source_VLA_2024.md
RAG_Retrieval_Augmented_Generation_2020.md
Context_Aware_RAG_VLA_2025.md
```

---

## 🔍 검색 및 참조

### 빠른 찾기
- **Ctrl+Shift+F**: 전체 폴더에서 키워드 검색
- **태그 활용**: `#critical`, `#rag`, `#latest` 등
- **README 참조**: 각 폴더의 README에서 논문 목록 확인

### 관련 논문 연결
- 각 논문 요약에서 **Related Work** 섹션 활용
- **Connected Papers** 도구로 논문 관계도 생성
- **인용 관계** 추적으로 핵심 논문들 발견

---

## 📈 진도 관리

### 읽기 상태 표시
각 논문 파일에 다음 상태 표시:
```markdown
**Status**: 📋 Queue / 📖 Reading / ✅ Done / 🔄 Review
**Priority**: 🔥 Critical / 📖 Important / 📚 Reference  
**Difficulty**: 🟢 Beginner / 🟡 Intermediate / 🔴 Advanced
**Last Read**: YYYY-MM-DD
```

### 진도 추적
- **주간 목표**: 폴더별 읽기 목표 설정
- **체크리스트**: 필수 논문들 완료 여부 확인
- **리뷰 주기**: 중요한 논문은 정기적으로 재독

---

## 🏷️ 태그 시스템

### 주요 태그들
```python
tags = {
    # 연구 영역
    "#vla": "Vision-Language-Action 관련",
    "#rag": "Retrieval-Augmented Generation",
    "#robotics": "로보틱스 기초",
    "#transformer": "Transformer 아키텍처",
    
    # 우선순위  
    "#critical": "필수 읽기",
    "#important": "중요함",
    "#reference": "참고용",
    
    # 활용도
    "#implementation": "구현 관련",
    "#theory": "이론 중심", 
    "#benchmark": "평가/벤치마크",
    "#survey": "서베이 논문",
    
    # 시기
    "#latest": "2024-2025 최신",
    "#classic": "정립된 이론",
    "#trend": "새로운 트렌드"
}
```

---

## 🤝 협업 및 공유

### 논문 공유
- **중요한 발견**: 새로운 인사이트나 핵심 아이디어
- **구현 아이디어**: 코드로 구현할 만한 내용
- **연구 연결**: 내 연구 주제와의 관련성

### 토론 포인트
- **비판적 분석**: 논문의 강점과 약점
- **개선 아이디어**: 더 나은 접근 방법
- **응용 가능성**: 다른 분야로의 확장

---

## 📅 정기 관리

### 매주
- [ ] 새로운 arXiv 논문 체크
- [ ] 읽은 논문들 정리 및 태그 정리
- [ ] 진도 상황 점검

### 매월  
- [ ] 폴더 구조 정리
- [ ] 중요도 재평가
- [ ] 새로운 트렌드 반영

---

## 🎯 최종 목표

**8주 후 달성 목표**:
- ✅ VLA 분야 핵심 논문 30편+ 완독
- ✅ 각 논문의 핵심 아이디어와 연결고리 파악
- ✅ 연구 제안서 작성을 위한 충분한 배경지식
- ✅ 최신 동향과 미래 연구 방향 이해

---

**이제 체계적인 논문 아카이브가 준비되었습니다!** 

각 폴더에 핵심 논문들을 정리해서 추가해드릴까요? 🚀

---

*Created: 2025-08-24*  
*Purpose: Systematic VLA research paper collection*  
*Target: 8-week intensive study program*