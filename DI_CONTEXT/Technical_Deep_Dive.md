# 🔬 기술적 심화 내용 - 교수님 질문 대비
## Selective RAG 구현의 기술적 세부사항

---

## 📐 **Confidence Estimation 구현 방법**

### **방법 1: Ensemble 기반**
```python
class EnsembleConfidence:
    def __init__(self, n_models=3):
        self.models = [VLAModel() for _ in range(n_models)]
    
    def estimate_confidence(self, observation):
        predictions = [m(observation) for m in self.models]
        
        # 분산이 크면 불확실
        variance = np.var(predictions, axis=0)
        confidence = 1.0 - normalize(variance)
        
        return confidence
```

### **방법 2: MC Dropout**
```python
class MCDropoutConfidence:
    def estimate_confidence(self, observation, n_samples=10):
        # Training mode로 전환 (dropout 활성화)
        self.model.train()
        
        predictions = []
        for _ in range(n_samples):
            pred = self.model(observation)
            predictions.append(pred)
        
        # 예측의 일관성 = 확신도
        uncertainty = np.std(predictions)
        confidence = 1.0 - uncertainty
        
        return confidence
```

### **방법 3: Output Distribution 분석**
```python
class DistributionConfidence:
    def estimate_confidence(self, observation):
        logits = self.model.get_logits(observation)
        probs = F.softmax(logits, dim=-1)
        
        # 엔트로피가 낮으면 확실
        entropy = -torch.sum(probs * torch.log(probs))
        confidence = 1.0 - (entropy / max_entropy)
        
        return confidence
```

---

## 🔍 **RAG 검색 최적화 전략**

### **1. 병렬 처리 구조**
```python
async def parallel_inference(observation):
    # 동시에 시작
    action_future = asyncio.create_task(
        generate_action(observation)
    )
    memory_future = asyncio.create_task(
        search_memory_if_needed(observation)
    )
    
    # 먼저 액션 받기
    action = await action_future
    confidence = estimate_confidence(action)
    
    if confidence < 0.7:
        # 메모리 검색 결과 기다리기
        memory = await memory_future
        action = refine_with_memory(action, memory)
    
    return action  # 대부분 경우 빠르게 리턴
```

### **2. 계층적 메모리 구조**
```python
class HierarchicalMemory:
    def __init__(self):
        self.cache = {}  # L1: 자주 쓰는 것
        self.recent = deque(maxlen=100)  # L2: 최근 것
        self.database = VectorDB()  # L3: 전체
    
    def search(self, query, confidence):
        # Confidence 낮을수록 깊게 검색
        if confidence > 0.5:
            return self.cache.get(query)
        elif confidence > 0.3:
            return self.search_recent(query)
        else:
            return self.search_all(query)
```

---

## 📊 **성능 최적화 기법**

### **1. Dynamic Batching**
```python
class DynamicBatcher:
    def process(self, requests):
        # Confidence별로 그룹화
        high_conf = [r for r in requests if r.conf > 0.7]
        low_conf = [r for r in requests if r.conf <= 0.7]
        
        # 고신뢰도는 바로 처리
        fast_results = batch_process(high_conf)
        
        # 저신뢰도는 RAG 포함 처리
        slow_results = batch_process_with_rag(low_conf)
        
        return merge(fast_results, slow_results)
```

### **2. Adaptive Threshold**
```python
class AdaptiveThreshold:
    def __init__(self):
        self.threshold = 0.7
        self.performance_history = []
    
    def update(self, was_correct, confidence):
        # 틀렸는데 confidence 높았으면 threshold 올리기
        if not was_correct and confidence > self.threshold:
            self.threshold += 0.05
        
        # 맞았는데 RAG 썼으면 threshold 낮추기
        elif was_correct and confidence < self.threshold:
            self.threshold -= 0.05
        
        self.threshold = np.clip(self.threshold, 0.3, 0.9)
```

---

## 🧪 **실험 설계**

### **1. Baseline 비교**
```python
baselines = {
    "No_RAG": "π₀ 스타일 (빠르지만 학습 없음)",
    "Always_RAG": "ELLMER 스타일 (느리지만 정확)",
    "Random_RAG": "랜덤하게 50% 검색",
    "Ours": "Confidence 기반 선택적"
}

metrics = {
    "speed": "Hz (추론 속도)",
    "accuracy": "성공률 %",
    "failure_repeat": "같은 실패 반복 횟수",
    "memory_usage": "MB"
}
```

### **2. Ablation Study**
```python
ablation = {
    "confidence_방법별": [
        "Ensemble",
        "MC Dropout",
        "Distribution"
    ],
    "threshold별": [0.5, 0.6, 0.7, 0.8, 0.9],
    "memory_size별": ["10MB", "100MB", "1GB"]
}
```

---

## 💾 **메모리 관리 전략**

### **Failure-Centric Storage**
```python
class FailureMemory:
    def should_store(self, episode):
        # 실패만 저장
        if episode.success:
            return False
        
        # 새로운 실패 패턴인지 확인
        if self.is_novel_failure(episode):
            return True
        
        # 반복되는 실패면 카운트만 증가
        self.increment_failure_count(episode)
        return False
    
    def compress(self, episode):
        # 핵심 정보만 추출
        return {
            "state": episode.critical_state,
            "action": episode.failed_action,
            "lesson": extract_lesson(episode)
        }
```

---

## 🎓 **이론적 배경**

### **Information Theory 관점**
```
H(action|observation) = Uncertainty

If H > threshold:
    I(action; memory) = Information gain from memory
    Use RAG when I > cost(retrieval)
```

### **Decision Theory 관점**
```
Expected Utility = P(success|no_RAG) × U(fast) 
                  + P(success|RAG) × U(accurate)

Optimize threshold to maximize EU
```

---

## 🔗 **관련 연구 연결**

### **Uncertainty in Deep Learning**
- Gal & Ghahramani (2016): Dropout as Bayesian Approximation
- Lakshminarayanan (2017): Simple and Scalable Uncertainty

### **Selective Computation**
- Graves (2016): Adaptive Computation Time
- Shazeer (2017): Mixture of Experts

### **Memory in RL**
- Pritzel (2017): Neural Episodic Control
- Fortunato (2019): Generalization in RL with Memory

---

## 🚨 **예상되는 기술적 챌린지**

### **1. Confidence Calibration**
```python
문제 = "모델이 과신할 수 있음"
해결 = "Temperature scaling, Platt scaling"
```

### **2. Distribution Shift**
```python
문제 = "훈련/테스트 분포 차이"
해결 = "Online adaptation, Continual learning"
```

### **3. Memory Explosion**
```python
문제 = "메모리 계속 증가"
해결 = "Forgetting mechanism, Importance sampling"
```

---

## 💬 **깊은 기술 질문 대비**

### **Q: "Gradient가 어떻게 흐르나요?"**
```
Confidence estimator는 별도 모듈로 훈련
Main policy는 RL/IL로 훈련
RAG는 non-differentiable (REINFORCE 가능)
```

### **Q: "Real-time constraint는?"**
```
Worst case: 50ms (20Hz)
Average case: 25ms (40Hz)
병렬 처리로 latency hiding
```

### **Q: "Sim2Real gap은?"**
```
Confidence가 Sim2Real indicator 역할
Real에서 confidence 낮으면 더 조심
Domain randomization으로 robust하게
```

---

*이 문서로 기술적 깊이를 보여주세요! 💪*