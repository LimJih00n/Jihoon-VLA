# 📄 Episodic Memory Verbalization using Hierarchical Representations
## 계층적 메모리로 로봇의 평생 경험을 효율적으로 관리

---

## 🎯 한 문장 요약
> **"로봇의 평생 경험을 트리 구조로 계층화하여 필요한 기억만 선택적으로 검색하는 시스템"**

---

## 📋 기본 정보
- **저자**: arXiv 2024 팀
- **발표**: arXiv 2024.09 (Sept 2024)
- **논문 링크**: [https://arxiv.org/abs/2409.17702](https://arxiv.org/abs/2409.17702)
- **프로젝트 페이지**: [https://hierarchical-emv.github.io/](https://hierarchical-emv.github.io/)
- **핵심 기여**: 계층적 메모리 구조로 선택적 검색 실현

---

## ❓ 해결하려는 문제

### 로봇 메모리의 딜레마
```python
memory_problems = {
    "정보_과부하": "모든 경험 저장시 폭발적 증가",
    "검색_비효율": "관련 경험 찾기 어려움",
    "중요도_구분": "핵심 vs 사소한 경험 구분 없음",
    "평생_학습": "오래된 기억 관리 어려움"
}
```

### 인간 기억의 교훈
- 인간은 **계층적으로** 기억
- **중요한 것만** 장기 기억
- **연관성**으로 검색

---

## 💡 핵심 아이디어: Hierarchical Episodic Memory

### 1. 트리 구조 메모리
```python
class HierarchicalMemory:
    def __init__(self):
        self.memory_tree = {
            "level_0": {  # Raw 센서 데이터 (1초)
                "type": "raw_perception",
                "retention": "1분",
                "size": "100MB/초"
            },
            "level_1": {  # 이벤트 (1분)
                "type": "event_summary",
                "retention": "1시간",
                "size": "1MB/분",
                "example": "컵을 잡음"
            },
            "level_2": {  # 태스크 (10분)
                "type": "task_sequence",
                "retention": "1일",
                "size": "100KB/태스크",
                "example": "커피 만들기"
            },
            "level_3": {  # 추상 개념 (1일)
                "type": "abstract_knowledge",
                "retention": "영구",
                "size": "10KB/개념",
                "example": "뜨거운 물 조심"
            }
        }
```

### 2. 선택적 압축 및 추상화
```python
def compress_memory(self, raw_experience):
    """하위 레벨을 상위 레벨로 추상화"""
    
    # Level 0 → Level 1: 이벤트 추출
    if is_significant(raw_experience):
        event = extract_event(raw_experience)
        self.level_1.append(event)
    
    # Level 1 → Level 2: 태스크 요약
    if task_completed(self.level_1):
        task_summary = summarize_task(self.level_1[-10:])
        self.level_2.append(task_summary)
        
    # Level 2 → Level 3: 교훈 추출
    if failure_detected(self.level_2):
        lesson = extract_lesson(self.level_2)
        self.level_3.append(lesson)  # 영구 저장!
    
    # 오래된 하위 레벨 삭제
    self.prune_old_memories()
```

### 3. LLM 기반 동적 검색
```python
class MemoryRetrieval:
    def __init__(self):
        self.llm = LanguageModel()
        
    def search_memory(self, query, context):
        """쿼리에 따라 적절한 레벨 검색"""
        
        # Step 1: 쿼리 분석
        query_type = self.llm.analyze(query)
        
        # Step 2: 레벨 결정
        if "구체적 행동" in query_type:
            search_level = "level_1"  # 최근 이벤트
        elif "과거 실패" in query_type:
            search_level = "level_3"  # 추상 교훈
        else:
            search_level = "level_2"  # 태스크 레벨
        
        # Step 3: 동적 트리 확장
        relevant_nodes = []
        current_node = self.memory_tree[search_level]
        
        while not enough_info(relevant_nodes):
            # 필요시 하위 레벨로 확장
            if needs_detail(query, current_node):
                current_node = expand_to_lower_level(current_node)
            relevant_nodes.append(current_node)
        
        return relevant_nodes
```

---

## 🔬 실험 결과

### 메모리 효율성
```python
memory_efficiency = {
    "기존_방식": {
        "1일_데이터": "8.6GB",
        "검색_시간": "2.3초",
        "정확도": "62%"
    },
    "계층적_메모리": {
        "1일_데이터": "120MB",  # 98% 감소!
        "검색_시간": "0.15초",  # 15배 빠름!
        "정확도": "84%"  # 더 정확!
    }
}
```

### 질의응답 성능
```python
qa_performance = {
    "단순_질문": {
        "예시": "마지막으로 컵을 잡은 때는?",
        "응답_시간": "0.1초",
        "정확도": "95%"
    },
    "복잡_추론": {
        "예시": "왜 어제 커피 만들기를 실패했나?",
        "응답_시간": "0.3초",
        "정확도": "88%"
    },
    "교훈_검색": {
        "예시": "뜨거운 물 다룰 때 주의점은?",
        "응답_시간": "0.05초",
        "정확도": "92%"
    }
}
```

---

## 🚀 혁신적 기여

### 1. **적응적 망각 (Adaptive Forgetting)**
```python
forgetting_policy = {
    "중요도_높음": "영구 보존 (실패, 성공 패턴)",
    "중요도_중간": "1일 보존 (일반 태스크)",
    "중요도_낮음": "1시간 보존 (반복 작업)",
    "raw_data": "1분 보존 (센서 데이터)"
}
```

### 2. **계층 간 연결**
- 상위 레벨이 하위 레벨 포인터 유지
- 필요시만 상세 정보 접근
- 메모리-속도 최적 균형

### 3. **LLM 통합 검색**
- 자연어 쿼리 이해
- 적절한 추상화 레벨 선택
- 동적 트리 탐색

---

## 💭 우리 연구와의 완벽한 시너지

### Episodic Memory + Adaptive RAG
```python
our_combined_system = {
    "계층적_저장": {
        "실패_경험": "Level 3 (영구)",
        "성공_패턴": "Level 2 (1일)",
        "일반_작업": "Level 1 (1시간)"
    },
    
    "선택적_검색": {
        "Confidence < 0.5": "Level 3 검색 (교훈)",
        "Confidence < 0.7": "Level 2 검색 (유사 태스크)",
        "Confidence > 0.7": "검색 스킵"
    },
    
    "메모리_관리": {
        "자동_압축": "하위→상위 추상화",
        "선택적_망각": "중요도 기반",
        "효율적_인덱싱": "계층별 별도 관리"
    }
}
```

### 우리가 직접 활용할 점
✅ **계층적 구조로 효율적 관리**
✅ **중요도 기반 선택적 저장**
✅ **LLM 통합 동적 검색**

### 우리가 추가할 점
✅ **Confidence 기반 트리거**
✅ **실패 우선 저장 정책**
✅ **실시간 제약 하 검색**

---

## 📝 교수님께 할 질문

1. "계층적 메모리의 레벨을 Confidence와 연동하면 더 효율적인 검색이 가능할까요?"

2. "실패 경험을 자동으로 Level 3로 승격시키는 메커니즘은 어떨까요?"

3. "트리 탐색 깊이를 실시간 제약에 맞춰 동적으로 조절하는 방법은?"

---

## 🎓 핵심 인사이트

> **"모든 것을 기억할 필요는 없다."**
> **중요한 것만 오래 기억하고,**
> **필요할 때만 깊이 검색한다.**
> **이것이 진정한 Selective Memory다.**

---

## 🔗 실제 구현 예시

### 커피 만들기 실패 시나리오
```python
# 실패 발생
failure_event = {
    "time": "2024-09-01 10:30",
    "task": "커피 만들기",
    "error": "뜨거운 물 쏟음",
    "cause": "컵 각도 잘못됨"
}

# 계층적 저장
memory.level_0.append(raw_sensor_data)  # 1분 후 삭제
memory.level_1.append("물 쏟음")        # 1시간 보존
memory.level_2.append("커피 실패")       # 1일 보존
memory.level_3.append("각도 주의")       # 영구 보존!

# 다음번 커피 만들 때
if confidence < 0.7:
    lesson = memory.search("커피 + 뜨거운물")
    # → Level 3에서 "각도 주의" 즉시 검색
    adjust_gripper_angle(lesson)
```

---

*이 논문은 선택적 메모리 관리의 이상적 모델을 제시하며, 우리의 Adaptive RAG와 결합시 완벽한 메모리 시스템 구축 가능*