# 🔍 Retrieval Systems 상세 설명

## 📌 개요
Retrieval System은 대규모 정보 저장소에서 관련 정보를 효율적으로 검색하는 시스템입니다. VLA에서는 과거 경험, 학습된 스킬, 유사한 상황을 빠르게 찾아 현재 작업에 활용하는 핵심 컴포넌트로, RAG(Retrieval-Augmented Generation) 패턴의 기반이 됩니다.

## 🎯 핵심 개념

### 1. 검색의 패러다임 변화

#### 전통적 검색 vs 의미적 검색
| 구분 | 전통적 검색 | 의미적 검색 |
|------|------------|------------|
| **매칭 방식** | 키워드 정확 매칭 | 의미 유사도 |
| **표현** | 단어/토큰 | 벡터 임베딩 |
| **장점** | 정확, 해석 가능 | 유연, 문맥 이해 |
| **단점** | 동의어 인식 불가 | 계산 비용 |

#### 검색 품질 요소
```
Relevance (관련성) = Semantic Similarity × Freshness × Authority
```

### 2. Dense Retrieval 원리

#### 벡터 임베딩
```python
text → encoder → d-dimensional vector
"빨간 공" → [0.2, -0.5, 0.8, ...]
"붉은 구체" → [0.19, -0.48, 0.79, ...]
```

#### 유사도 메트릭
1. **Cosine Similarity**
   ```
   sim(a,b) = (a·b) / (||a|| × ||b||)
   범위: [-1, 1], 1에 가까울수록 유사
   ```

2. **Euclidean Distance**
   ```
   dist(a,b) = √Σ(aᵢ - bᵢ)²
   범위: [0, ∞), 0에 가까울수록 유사
   ```

3. **Dot Product**
   ```
   score(a,b) = Σ(aᵢ × bᵢ)
   범위: (-∞, ∞), 클수록 유사
   ```

### 3. Sparse Retrieval 원리

#### BM25 (Best Matching 25)
```
score(q,d) = Σ IDF(qᵢ) × (f(qᵢ,d) × (k₁+1)) / (f(qᵢ,d) + k₁×(1-b+b×|d|/avgdl))
```

구성 요소:
- **TF (Term Frequency)**: 문서 내 단어 빈도
- **IDF (Inverse Document Frequency)**: 단어의 희귀성
- **문서 길이 정규화**: 긴 문서 불이익 방지

#### TF-IDF
```
TF-IDF(t,d,D) = TF(t,d) × IDF(t,D)
where IDF(t,D) = log(|D| / |{d∈D: t∈d}|)
```

### 4. Vector Database 구조

#### 인덱스 유형

1. **Flat Index**
   - 선형 탐색
   - 정확하지만 느림
   - O(n) 복잡도

2. **IVF (Inverted File)**
   - 클러스터링 기반
   - Voronoi cells
   - O(√n) 복잡도

3. **HNSW (Hierarchical Navigable Small World)**
   - 그래프 기반
   - 계층적 구조
   - O(log n) 복잡도

4. **LSH (Locality Sensitive Hashing)**
   - 해시 기반
   - 확률적 근사
   - O(1) 복잡도

## 🏗️ 구현 메커니즘 상세

### 1. 효율적인 벡터 검색

#### FAISS (Facebook AI Similarity Search)
```python
import faiss
import numpy as np

# Index types
index_flat = faiss.IndexFlatL2(d)  # Exact search
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)  # Approximate
index_hnsw = faiss.IndexHNSWFlat(d, M)  # Graph-based

# GPU acceleration
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
```

#### Annoy (Approximate Nearest Neighbors Oh Yeah)
```python
from annoy import AnnoyIndex

index = AnnoyIndex(f, 'angular')  # or 'euclidean'
for i, v in enumerate(vectors):
    index.add_item(i, v)
index.build(n_trees)  # More trees = better accuracy
```

### 2. Hybrid Retrieval 전략

#### Score Fusion
```python
def reciprocal_rank_fusion(rankings, k=60):
    """RRF: 여러 랭킹 결합"""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

#### Late Interaction
```python
class ColBERT:
    """Late interaction for efficient retrieval"""
    def score(query_embeds, doc_embeds):
        # MaxSim: max similarity for each query token
        scores = []
        for q_emb in query_embeds:
            max_sim = max(cosine_sim(q_emb, d_emb) for d_emb in doc_embeds)
            scores.append(max_sim)
        return sum(scores)
```

### 3. 학습 가능한 검색

#### Dense Passage Retrieval (DPR)
```python
class DPR:
    def __init__(self):
        self.query_encoder = BertModel()
        self.passage_encoder = BertModel()
    
    def train(self, queries, positive_passages, negative_passages):
        q_emb = self.query_encoder(queries)
        pos_emb = self.passage_encoder(positive_passages)
        neg_emb = self.passage_encoder(negative_passages)
        
        # Contrastive loss
        pos_scores = (q_emb * pos_emb).sum(dim=-1)
        neg_scores = (q_emb * neg_emb).sum(dim=-1)
        loss = -log(exp(pos_scores) / (exp(pos_scores) + Σexp(neg_scores)))
```

#### Cross-Encoder Reranking
```python
class CrossEncoder:
    """정밀 재순위화"""
    def rerank(self, query, candidates):
        scores = []
        for doc in candidates:
            input_text = f"{query} [SEP] {doc}"
            score = self.model(input_text)
            scores.append(score)
        return sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
```

## 🤖 VLA에서의 검색 시스템

### 1. Experience Retrieval

#### 에피소드 검색
```python
class EpisodeRetrieval:
    def __init__(self):
        self.episode_encoder = TransformerEncoder()
        self.index = faiss.IndexFlatIP(768)
    
    def encode_episode(self, states, actions, rewards):
        # Temporal encoding
        trajectory = torch.stack([s for s in states])
        encoded = self.episode_encoder(trajectory)
        
        # Add outcome information
        success_weight = rewards.sum() / len(rewards)
        weighted = encoded * success_weight
        
        return weighted.mean(dim=0)  # Aggregate
```

#### 서브골 검색
```python
def retrieve_subgoals(current_state, goal, subgoal_library):
    """중간 목표 검색"""
    # Encode current and goal
    gap_vector = goal_encoder(goal) - state_encoder(current_state)
    
    # Find similar gaps in library
    similar_gaps = subgoal_library.search(gap_vector, k=5)
    
    # Extract subgoal sequence
    subgoals = []
    for gap in similar_gaps:
        subgoals.extend(gap['subgoal_sequence'])
    
    return deduplicate_and_order(subgoals)
```

### 2. Skill Library Management

#### 계층적 스킬 검색
```python
class HierarchicalSkillLibrary:
    def __init__(self):
        self.primitive_skills = VectorDatabase()  # 기본 동작
        self.composite_skills = VectorDatabase()  # 복합 동작
        self.meta_skills = VectorDatabase()      # 추상 전략
    
    def retrieve_skill_chain(self, task_description):
        # Top-down retrieval
        meta = self.meta_skills.search(task_description, k=1)[0]
        composites = []
        for comp_id in meta['composite_ids']:
            comp = self.composite_skills.get(comp_id)
            composites.append(comp)
        
        primitives = []
        for comp in composites:
            for prim_id in comp['primitive_ids']:
                prim = self.primitive_skills.get(prim_id)
                primitives.append(prim)
        
        return {'meta': meta, 'composites': composites, 'primitives': primitives}
```

#### 스킬 적응
```python
def adapt_skill(retrieved_skill, current_context):
    """검색된 스킬을 현재 상황에 맞게 조정"""
    # Parameter adaptation
    adapted_params = skill_adapter(
        skill_params=retrieved_skill['parameters'],
        context=current_context
    )
    
    # Precondition checking
    if not check_preconditions(retrieved_skill['preconditions'], current_context):
        # Find alternative or modify
        adapted_params = modify_for_preconditions(adapted_params, current_context)
    
    return adapted_params
```

### 3. Multi-Modal Retrieval

#### Cross-Modal Alignment
```python
class CrossModalRetriever:
    def __init__(self):
        self.vision_proj = nn.Linear(2048, 512)
        self.language_proj = nn.Linear(768, 512)
        self.action_proj = nn.Linear(7, 512)
    
    def align_modalities(self, vision, language, action):
        # Project to common space
        v_emb = self.vision_proj(vision)
        l_emb = self.language_proj(language)
        a_emb = self.action_proj(action)
        
        # Normalize for cosine similarity
        v_emb = F.normalize(v_emb, dim=-1)
        l_emb = F.normalize(l_emb, dim=-1)
        a_emb = F.normalize(a_emb, dim=-1)
        
        return v_emb, l_emb, a_emb
```

#### Attention-based Retrieval
```python
class AttentiveRetrieval:
    def retrieve_with_attention(self, query, database, attention_weights):
        """주의 기반 검색"""
        # Weight different aspects
        weighted_query = query * attention_weights
        
        # Multi-head retrieval
        results = []
        for head in range(self.num_heads):
            head_query = self.head_projection[head](weighted_query)
            head_results = database.search(head_query, k=3)
            results.extend(head_results)
        
        # Aggregate and deduplicate
        return self.aggregate_results(results)
```

## 🔬 고급 기법

### 1. Learned Index

#### Neural Index
```python
class LearnedIndex:
    """학습된 인덱스 구조"""
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Position prediction
        )
    
    def train(self, vectors, positions):
        # Learn to predict storage position
        for v, pos in zip(vectors, positions):
            pred_pos = self.model(v)
            loss = F.mse_loss(pred_pos, pos)
            loss.backward()
    
    def search(self, query):
        # Predict position range
        pred_pos = self.model(query)
        search_range = [pred_pos - ε, pred_pos + ε]
        return linear_search_in_range(search_range)
```

### 2. Continual Learning in Retrieval

#### Dynamic Index Update
```python
class DynamicRetriever:
    def __init__(self):
        self.index = None
        self.buffer = []
        self.rebuild_threshold = 1000
    
    def add_incremental(self, vectors, metadata):
        self.buffer.extend(zip(vectors, metadata))
        
        if len(self.buffer) > self.rebuild_threshold:
            self.rebuild_index()
    
    def rebuild_index(self):
        """효율적 인덱스 재구축"""
        # Merge buffer with existing
        all_vectors = self.get_all_vectors()
        
        # Clustering for IVF
        kmeans = faiss.Kmeans(d, ncentroids)
        kmeans.train(all_vectors)
        
        # Build new index
        self.index = faiss.IndexIVFFlat(kmeans.index, d, ncentroids)
        self.index.add(all_vectors)
        
        self.buffer.clear()
```

### 3. Privacy-Preserving Retrieval

#### Differential Privacy
```python
def dp_retrieval(query, database, epsilon=1.0):
    """차등 프라이버시 검색"""
    # Add noise to query
    noise = np.random.laplace(0, 1/epsilon, query.shape)
    noisy_query = query + noise
    
    # Retrieve with noisy query
    results = database.search(noisy_query, k=10)
    
    # Add noise to scores
    for r in results:
        r['score'] += np.random.laplace(0, 1/epsilon)
    
    return results
```

## 💡 실전 최적화 가이드

### 1. 인덱스 선택 가이드

| 데이터 크기 | 정확도 요구 | 추천 인덱스 | 이유 |
|------------|------------|------------|------|
| < 1K | 높음 | Flat | 정확, 빠름 |
| 1K-10K | 중간 | IVF | 균형 |
| 10K-100K | 중간 | HNSW | 확장성 |
| > 100K | 낮음 | LSH | 메모리 효율 |

### 2. 성능 튜닝

#### Batch Processing
```python
def batch_search(queries, index, batch_size=100):
    results = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        batch_results = index.search(batch, k=10)
        results.extend(batch_results)
    return results
```

#### Caching Strategy
```python
class LRUCache:
    def __init__(self, capacity=1000):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

### 3. 품질 평가

#### 검색 품질 메트릭
```python
def evaluate_retrieval_quality(ground_truth, predictions):
    metrics = {}
    
    # Precision@K
    metrics['p@5'] = precision_at_k(ground_truth, predictions, k=5)
    
    # Mean Reciprocal Rank
    metrics['mrr'] = mean_reciprocal_rank(ground_truth, predictions)
    
    # Normalized Discounted Cumulative Gain
    metrics['ndcg'] = ndcg(ground_truth, predictions)
    
    # Semantic Similarity
    metrics['semantic_sim'] = average_semantic_similarity(ground_truth, predictions)
    
    return metrics
```

## 🚀 최신 연구 동향

### 1. Neural Architecture
- **RETRO**: Retrieval-Enhanced Transformer
- **REALM**: Retrieval-Augmented Language Model
- **RAG**: Retrieval-Augmented Generation

### 2. Efficient Retrieval
- **SCANN**: Scalable Nearest Neighbors
- **DiskANN**: Billion-scale disk-based index
- **Pinecone**: Managed vector database

### 3. Multimodal Retrieval
- **CLIP**: Contrastive Language-Image Pre-training
- **ALIGN**: Large-scale noisy image-text pairs
- **Flamingo**: Few-shot multimodal learning

## ⚠️ 주의사항 및 한계

### 주요 문제점
1. **Semantic Gap**: 임베딩이 의미를 완벽히 포착 못함
2. **Distribution Shift**: 학습/테스트 분포 차이
3. **Storage Overhead**: 대규모 인덱스 저장 비용
4. **Update Latency**: 실시간 업데이트 어려움

### 해결 방안
1. **Hybrid Methods**: Dense + Sparse 결합
2. **Continual Learning**: 점진적 업데이트
3. **Compression**: 벡터 양자화
4. **Distributed Systems**: 분산 저장/검색

## 📚 추가 학습 자료

### 핵심 논문
- "Dense Passage Retrieval for Open-Domain Question Answering"
- "ColBERT: Efficient and Effective Passage Search"
- "Approximate Nearest Neighbor Search in High Dimensions"

### 도구 및 라이브러리
- FAISS: 벡터 검색
- Elasticsearch: 텍스트 검색
- Weaviate: 벡터 데이터베이스
- Milvus: 확장 가능한 벡터 DB

### 벤치마크
- MS MARCO: Passage retrieval
- BEIR: Zero-shot retrieval
- TREC: Text REtrieval Conference

## 🎯 핵심 요약

Retrieval System은 VLA가 방대한 경험과 지식을 효율적으로 활용할 수 있게 하는 핵심 인프라입니다. Dense Retrieval의 의미적 이해력과 Sparse Retrieval의 정확성을 결합한 Hybrid 접근, 효율적인 Vector Database 구축, RAG 패턴의 활용이 성공적인 시스템 구현의 열쇠입니다. 실시간 성능, 확장성, 검색 품질의 균형을 맞추는 것이 중요합니다.