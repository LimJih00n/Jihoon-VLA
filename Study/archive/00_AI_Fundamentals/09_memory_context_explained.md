# 🧠 Memory & Context 상세 설명

## 📌 개요
Memory와 Context 시스템은 AI가 과거 정보를 저장하고 활용하여 현재 상황을 이해하고 미래 행동을 계획하는 핵심 메커니즘입니다. VLA에서는 복잡한 작업 수행, 장기 의존성 처리, 경험 기반 학습을 가능하게 합니다.

## 🎯 핵심 개념

### 1. 메모리 시스템의 계층 구조

#### 인간 기억 vs AI 메모리
| 인간 기억 | AI 메모리 | 특징 | VLA 활용 |
|-----------|-----------|------|----------|
| **감각 기억** | Attention Cache | 1-2초, 원시 정보 | 즉각적인 센서 입력 |
| **작업 기억** | Working Memory | 7±2 항목, 활성 처리 | 현재 작업 상태 |
| **장기 기억** | External Memory | 무제한, 영구 저장 | 학습된 스킬, 경험 |

#### 메모리 용량과 지속 시간
```
Sensory: ~1초, 대용량 → 필터링 필요
Working: ~30초, 7±2 chunks → 선택적 저장
Long-term: 영구, 무제한 → 효율적 검색 필요
```

### 2. Working Memory 원리

#### Miller's Magic Number 7±2
인간의 작업 기억 용량 한계를 AI에 적용:
- **Chunking**: 정보를 의미 단위로 묶기
- **Rehearsal**: 반복을 통한 유지
- **Central Executive**: 주의 할당 제어

#### Neural Implementation
```python
# Slot-based memory
memory_slots = [slot_1, slot_2, ..., slot_7]
attention_weights = softmax(query @ keys)
output = attention_weights @ values
```

### 3. Long-term Memory 메커니즘

#### Storage Mechanisms
1. **Hebbian Learning**: "함께 발화하는 뉴런은 함께 연결된다"
2. **Consolidation**: 단기 → 장기 전환
3. **Retrieval**: 연관 기반 검색

#### Memory Indexing
```python
# Content-based addressing
similarity = cosine_similarity(query, memory_bank)
retrieved = weighted_sum(memory_bank, similarity)

# Location-based addressing
address = hash(key) % memory_size
retrieved = memory_bank[address]
```

### 4. Episodic vs Semantic Memory

#### Episodic Memory (에피소드 기억)
- **What**: 특정 사건의 기억
- **When/Where**: 시공간 정보 포함
- **Example**: "어제 빨간 공을 집었다"

#### Semantic Memory (의미 기억)
- **What**: 일반적 지식
- **How**: 절차적 지식
- **Example**: "공은 둥글다"

## 🏗️ 구현 메커니즘 상세

### 1. Attention-based Working Memory

#### Key-Value Memory Networks
```python
class KeyValueMemory:
    def __init__(self):
        self.keys = []    # 메모리 주소
        self.values = []  # 메모리 내용
    
    def write(self, key, value):
        # Content-based addressing
        weights = softmax(key @ self.keys.T)
        # Weighted update
        self.values = (1-weights) * self.values + weights * value
    
    def read(self, query):
        weights = softmax(query @ self.keys.T)
        return weights @ self.values
```

#### Memory Networks Architecture
1. **Input Module**: 입력 인코딩
2. **Memory Module**: 저장 및 업데이트
3. **Output Module**: 응답 생성
4. **Response Module**: 최종 출력

### 2. Neural Turing Machine (NTM) 개념

#### 읽기 연산
```
r_t = Σ_i w_t(i) * M_t(i)
```
- w_t: 주의 가중치
- M_t: 메모리 매트릭스

#### 쓰기 연산
```
M_t(i) = M_{t-1}(i) * (1 - w_t(i) * e_t) + w_t(i) * a_t
```
- e_t: 지우기 벡터
- a_t: 추가 벡터

### 3. Hierarchical Temporal Memory

#### 시간적 계층 구조
```
Level 1: 밀리초 (센서 데이터)
    ↓ 추상화
Level 2: 초 단위 (동작 패턴)
    ↓ 추상화
Level 3: 분 단위 (작업 단계)
    ↓ 추상화
Level 4: 시간 단위 (전체 작업)
```

#### Sparse Distributed Representation
- 2% 활성 뉴런
- 높은 용량
- 노이즈 강건성

### 4. Context Management

#### Context Window
```python
class ContextWindow:
    def __init__(self, max_length=2048):
        self.max_length = max_length
        self.buffer = deque(maxlen=max_length)
    
    def update(self, token):
        self.buffer.append(token)
        if len(self.buffer) > self.max_length:
            self.buffer.popleft()  # FIFO
```

#### Hierarchical Context
```
Immediate: 최근 5 토큰
Local: 최근 50 토큰
Global: 전체 문서/에피소드
Meta: 작업 수준 컨텍스트
```

## 🤖 VLA에서의 메모리 활용

### 1. Task Memory

#### 작업 단계 기억
```python
task_memory = {
    "goal": "assemble product",
    "completed_steps": ["pick_part_A", "pick_part_B"],
    "current_step": "align_parts",
    "remaining_steps": ["insert_screw", "tighten"],
    "context": {"tool": "screwdriver", "location": "workbench"}
}
```

#### Procedural Memory
```python
skill_library = {
    "grasp": GraspSkill(),
    "insert": InsertSkill(),
    "rotate": RotateSkill()
}
```

### 2. Object-Centric Memory

#### Object Permanence
물체가 시야에서 사라져도 존재한다는 개념:
```python
object_memory = {
    "red_cube": {
        "last_seen": timestamp,
        "location": [x, y, z],
        "properties": {"color": "red", "shape": "cube"},
        "confidence": 0.95
    }
}
```

#### Spatial Memory
```python
spatial_map = {
    "workbench": {"objects": ["tool_1", "part_A"], "free_space": 0.3},
    "storage": {"objects": ["part_B", "part_C"], "free_space": 0.7}
}
```

### 3. Experience Replay

#### Prioritized Experience Replay
```python
priority = |TD_error| + ε
probability = priority^α / Σ priority^α
```

중요한 경험을 더 자주 재생

#### Hindsight Experience Replay
실패한 경험도 다른 목표 달성으로 재해석:
```python
if not achieved(goal):
    virtual_goal = final_state
    relabel_trajectory(trajectory, virtual_goal)
    store_as_success(trajectory)
```

## 🔬 고급 메모리 기법

### 1. Memory Consolidation

#### System Consolidation
```python
def consolidate_memory(short_term, long_term):
    # Replay during "sleep"
    for memory in short_term:
        if is_important(memory):
            compressed = compress(memory)
            long_term.store(compressed)
    
    # Synaptic consolidation
    strengthen_connections(frequently_accessed)
    weaken_connections(rarely_accessed)
```

#### Memory Compression
```python
def compress_episode(episode):
    # Extract key frames
    keyframes = detect_keyframes(episode)
    # Abstract representation
    abstract = encode_abstract(keyframes)
    # Store compressed
    return {"keyframes": keyframes, "abstract": abstract}
```

### 2. Associative Memory

#### Hopfield Networks
```python
class HopfieldMemory:
    def __init__(self, patterns):
        # Store patterns in weights
        self.W = sum([p @ p.T for p in patterns])
        np.fill_diagonal(self.W, 0)
    
    def recall(self, partial):
        # Iterative retrieval
        state = partial
        while not converged:
            state = sign(self.W @ state)
        return state
```

#### Content-Addressable Memory
```python
def content_addressable_retrieve(query, memory):
    # Retrieve by content, not address
    best_match = None
    max_similarity = -inf
    
    for item in memory:
        similarity = compute_similarity(query, item.content)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = item
    
    return best_match
```

### 3. Forgetting Mechanisms

#### Adaptive Forgetting
```python
def adaptive_forget(memory_bank, capacity):
    if len(memory_bank) > capacity:
        # Compute importance scores
        importance = []
        for memory in memory_bank:
            score = memory.access_frequency * memory.recency * memory.relevance
            importance.append(score)
        
        # Remove least important
        threshold = percentile(importance, 20)
        memory_bank = [m for m, i in zip(memory_bank, importance) if i > threshold]
    
    return memory_bank
```

#### Catastrophic Forgetting Prevention
```python
class ElasticWeightConsolidation:
    def __init__(self, model):
        self.model = model
        self.fisher_matrix = {}  # Importance of parameters
    
    def penalty(self):
        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_matrix:
                loss += (self.fisher_matrix[name] * 
                        (param - self.old_params[name])**2).sum()
        return loss
```

## 💡 실전 적용 가이드

### 1. 메모리 크기 선택

#### Working Memory
- **작은 작업**: 3-5 슬롯
- **복잡한 작업**: 7-10 슬롯
- **다중 작업**: 10-15 슬롯

#### Long-term Memory
- **제한된 도메인**: 100-1000 항목
- **개방형 도메인**: 10000+ 항목
- **증분 학습**: 동적 확장

### 2. 메모리 업데이트 전략

#### Write Strategies
1. **FIFO**: 가장 오래된 것 교체
2. **LRU**: 가장 적게 사용된 것 교체
3. **LFU**: 가장 적은 빈도 교체
4. **Priority**: 중요도 기반 교체

#### Read Strategies
1. **Exact Match**: 정확한 매칭
2. **Similarity**: 유사도 기반
3. **Temporal**: 시간 기반
4. **Associative**: 연관 기반

### 3. 성능 최적화

#### Memory Indexing
```python
class EfficientMemory:
    def __init__(self):
        self.index = faiss.IndexFlatL2(dim)  # Fast similarity search
        self.metadata = {}
    
    def add(self, vector, meta):
        idx = self.index.ntotal
        self.index.add(vector)
        self.metadata[idx] = meta
    
    def search(self, query, k=5):
        distances, indices = self.index.search(query, k)
        return [(self.metadata[i], d) for i, d in zip(indices[0], distances[0])]
```

#### Caching
```python
@lru_cache(maxsize=128)
def expensive_retrieval(query):
    return search_in_memory(query)
```

## 🚀 최신 연구 동향

### 1. Transformer Memory
- **Transformer-XL**: Segment-level recurrence
- **Compressive Transformer**: 압축된 메모리
- **∞-former**: Infinite memory transformer

### 2. Neuromorphic Memory
- **Spiking Neural Networks**: 생물학적 메모리
- **Memristor**: 하드웨어 메모리
- **Neuromorphic Chips**: 뇌 모방 칩

### 3. Quantum Memory
- **Quantum Associative Memory**: 양자 중첩
- **Quantum RAM**: 지수적 용량
- **Quantum Error Correction**: 노이즈 처리

## ⚠️ 주의사항 및 한계

### 주요 문제점
1. **Memory Overflow**: 용량 초과
2. **Memory Leak**: 불필요한 정보 축적
3. **Retrieval Failure**: 검색 실패
4. **Context Confusion**: 문맥 혼동

### 해결 방안
1. **Garbage Collection**: 주기적 정리
2. **Memory Compression**: 압축 저장
3. **Redundant Storage**: 중복 저장
4. **Error Correction**: 오류 수정

## 📚 추가 학습 자료

### 핵심 논문
- "Neural Turing Machines" (Graves et al.)
- "Memory Networks" (Weston et al.)
- "Differentiable Neural Computer" (Graves et al.)

### 구현 라이브러리
- Faiss (Facebook): 벡터 검색
- Annoy (Spotify): 근사 최근접 이웃
- HNSW: 계층적 탐색

### 벤치마크
- bAbI tasks: 추론과 메모리
- CLEVR: 시각적 추론
- Meta-World: 로봇 작업

## 🎯 핵심 요약

메모리와 컨텍스트 시스템은 VLA가 과거 경험을 활용하고 복잡한 작업을 수행할 수 있게 하는 핵심입니다. Working Memory는 즉각적인 작업 처리를, Long-term Memory는 지식 축적을, Episodic Memory는 경험 기반 학습을 가능하게 합니다. 효율적인 저장, 검색, 업데이트 메커니즘과 함께 계층적 구조를 통해 다양한 시간 스케일의 정보를 통합 관리하는 것이 성공적인 구현의 열쇠입니다.