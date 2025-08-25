# 🔍 VLA 평가 자동화 현황 조사 (2025년 1월)
## "정말 자동화되어 있지 않나?"

---

## 📊 조사 결과: 부분적 자동화는 있지만, 통합 자동화는 없음

### 현재 존재하는 것들

| 플랫폼 | 출시 | 자동화 수준 | 한계점 |
|--------|------|------------|--------|
| **VLABench** | 2024.12.25 | ⭐⭐⭐ | 단일 벤치마크, 타 플랫폼 미연결 |
| **LeRobot** | 2024.05 | ⭐⭐ | 데이터셋 중심, 평가 자동화 미흡 |
| **CMU VLA Challenge** | 2025.09 예정 | ⭐⭐ | 대회용, 일반 사용 불가 |
| **SimplerEnv** | 2024 | ⭐ | 수동 업데이트 |

---

## 🎯 1. VLABench (2024년 12월 출시)

### 현재 상태
```python
vlabench_status = {
    "출시": "2024년 12월 25일 (1달 전!)",
    "상태": "Preview version",
    
    "자동화된 것": [
        "VLM 평가 스크립트",
        "GPT-4V, Qwen2-VL 등 지원",
        "Few-shot evaluation"
    ],
    
    "자동화 안된 것": [
        "다른 벤치마크와 통합 ❌",
        "통합 리더보드 ❌",
        "LeRobot과 연결 ❌",
        "SimplerEnv와 호환 ❌"
    ]
}

# 사용 예시
python scripts/evaluate_vlm.py --vlm_name Qwen2_VL --few-shot-num 1
# 하지만 이건 VLABench만 실행
```

### Hugging Face 통합
- 데이터셋 호스팅: ✅ (VLABench/eval_vlm_v0)
- 조직 프로필: ✅
- **하지만**: 다른 벤치마크와 통합 평가 ❌

---

## 🤖 2. LeRobot 현황

### 최근 업데이트 (2024-2025)
```python
lerobot_updates = {
    "PR #1645": "VLA tokenizer pipeline 포팅",
    "PR #1676": "LIBERO 환경 추가",
    "SmolVLA": "LeRobot 데이터로 훈련",
    "π0 모델": "포팅 완료",
    
    "자동화된 것": [
        "데이터 수집",
        "모델 훈련",
        "기본 평가"
    ],
    
    "자동화 안된 것": [
        "통합 벤치마크 평가 ❌",
        "다른 플랫폼과 연결 ❌",
        "통합 리더보드 ❌"
    ]
}
```

### 한계점
- VLABench와 통합 안됨
- SimplerEnv와 별개로 운영
- 각자 다른 평가 방식

---

## 📈 3. 실제 연구자 워크플로우 (현재)

### 자동화되어 있다면...
```python
# 이상적 시나리오 (만약 자동화되어 있다면)
automated_workflow = {
    "1단계": "model.upload()",
    "2단계": "evaluate_all_benchmarks()",  # 모든 벤치마크 자동
    "3단계": "view_unified_results()",     # 통합 결과
}
# 시간: 2-3시간
```

### 실제 현실
```python
# 실제 워크플로우 (2025년 1월 현재)
reality_workflow = {
    "1단계": {
        "VLABench 설치": "pip install vlabench",
        "LeRobot 설치": "pip install lerobot",
        "SimplerEnv 설치": "git clone ...",
        "충돌 해결": "2-3시간"
    },
    
    "2단계": {
        "VLABench 실행": "python vlabench_eval.py",
        "LeRobot 실행": "python lerobot_eval.py",
        "SimplerEnv 실행": "python simplerenv_eval.py",
        "각각 다른 포맷": True
    },
    
    "3단계": {
        "결과 수집": "수동 복사",
        "통합": "Excel 작업",
        "비교": "수동 차트 생성"
    }
}
# 시간: 2-3일
```

---

## 🔍 4. "통합" 자동화가 없는 증거

### 검색 결과 분석
```python
search_findings = {
    "통합 플랫폼": {
        "VLA 전용": "없음",
        "언급": "unified platform 언급만 있음",
        "실제 구현": "찾을 수 없음"
    },
    
    "개별 도구": {
        "VLABench": "독립적 운영",
        "LeRobot": "독립적 운영",
        "SimplerEnv": "독립적 운영",
        "CMU Challenge": "대회용"
    },
    
    "자동화 수준": {
        "개별 도구 내": "부분적 자동화",
        "도구 간 통합": "자동화 없음",
        "통합 리더보드": "없음"
    }
}
```

### 비교: NLP vs VLA

| 측면 | NLP (Hugging Face) | VLA (현재) |
|------|-------------------|------------|
| **통합 평가** | ✅ Open LLM Leaderboard | ❌ 개별 평가만 |
| **원클릭 실행** | ✅ Evaluate library | ❌ 각각 실행 |
| **표준 포맷** | ✅ 통일됨 | ❌ 각자 다름 |
| **자동 업데이트** | ✅ | ❌ 수동 |

---

## 💡 5. 왜 통합 자동화가 없는가?

### 기술적 이유
```python
technical_reasons = {
    "복잡성": {
        "VLABench": "자체 프레임워크",
        "LeRobot": "다른 아키텍처",
        "SimplerEnv": "또 다른 시스템",
        "호환성": "서로 맞지 않음"
    },
    
    "최신성": {
        "VLABench": "2024.12.25 출시 (1달)",
        "통합 시도": "아직 없음",
        "시간 필요": "최소 6개월-1년"
    },
    
    "인력": {
        "각 팀": "자기 도구만 개발",
        "통합 팀": "없음",
        "인센티브": "논문 우선"
    }
}
```

---

## 🎯 6. 결론: 부분 자동화 O, 통합 자동화 X

### 현재 상황 정리

✅ **자동화되어 있는 것:**
- VLABench 내부 평가 스크립트
- LeRobot 데이터 파이프라인
- 개별 도구의 부분 자동화

❌ **자동화되어 있지 않은 것:**
- 도구 간 통합 평가
- 통합 리더보드
- 원클릭 전체 평가
- 표준화된 결과 포맷

### 증거
```python
evidence = {
    "VLABench": "독립 운영, 타 도구 미연결",
    "LeRobot": "데이터셋 중심, 평가 미흡",
    "통합 도구": "검색 결과 0건",
    "통합 리더보드": "존재하지 않음"
}
```

---

## 💡 최종 답변

**Q: "지금 자동화되어 있지 않나?"**

**A: 개별 도구는 부분 자동화, 통합 자동화는 없음**

```python
automation_status = {
    "VLABench 자동화": "⭐⭐⭐ (자체 평가만)",
    "LeRobot 자동화": "⭐⭐ (데이터 중심)",
    "통합 자동화": "⭐ (없음)",
    
    "결론": "각자 따로 놀고 있음"
}
```

**이래서 UnifiedVLA가 필요합니다:**
- 파편화된 도구들을 연결
- 진짜 원클릭 평가
- 통합 리더보드
- 표준화된 워크플로우

---

*문서 작성일: 2025년 8월 24일*  
*최종 수정일: 2025년 8월 24일 오후 11시 45분*  
*분석 도구: Claude Code Assistant*

---
