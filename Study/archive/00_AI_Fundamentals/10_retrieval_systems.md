# 🔍 Retrieval Systems: 효율적인 정보 검색

**목표**: Vector Database, Dense/Sparse Retrieval, Semantic Search의 이해 및 VLA RAG 시스템 구축  
**시간**: 2-3시간  
**전제조건**: 01_neural_networks_basics.md, 02_attention_mechanism.md  

---

## 🎯 개발자를 위한 직관적 이해

### Retrieval이란?
```python
traditional_search = {
    "keyword_matching": "정확한 단어 매칭",
    "limitations": "동의어, 문맥 이해 불가"
}

semantic_retrieval = {
    "vector_similarity": "의미적 유사도 기반",
    "advantages": "동의어, 문맥, 의도 파악 가능"
}

# 예시
query = "빨간 공을 집어라"
relevant_memories = [
    "붉은색 구체를 잡았다",      # 의미적으로 유사
    "crimson sphere grasping",  # 다른 언어도 가능
    "적색 볼 파지 동작"         # 전문용어도 매칭
]
```

### VLA에서 왜 중요한가?
- **경험 재사용**: 과거 유사 상황에서 학습
- **효율성**: 모든 것을 기억할 필요 없음
- **일반화**: 유사한 태스크 간 지식 전이

---

## 🏗️ 기본 구조 및 구현

### 1. Dense Retrieval (벡터 기반)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

class DenseRetriever(nn.Module):
    """Dense vector 기반 검색 시스템"""
    def __init__(self, encoder_model, vector_dim=768, index_type='flat'):
        super().__init__()
        self.encoder = encoder_model
        self.vector_dim = vector_dim
        self.index_type = index_type
        
        # Vector storage
        self.vectors = []
        self.metadata = []
        
        # Optional: Use FAISS for efficient search
        self.use_faiss = False
        try:
            import faiss
            self.use_faiss = True
            if index_type == 'flat':
                self.index = faiss.IndexFlatL2(vector_dim)
            elif index_type == 'ivf':
                quantizer = faiss.IndexFlatL2(vector_dim)
                self.index = faiss.IndexIVFFlat(quantizer, vector_dim, 100)
            elif index_type == 'hnsw':
                self.index = faiss.IndexHNSWFlat(vector_dim, 32)
        except ImportError:
            print("FAISS not available, using numpy")
    
    def encode(self, texts):
        """텍스트를 벡터로 인코딩"""
        with torch.no_grad():
            if isinstance(texts, str):
                texts = [texts]
            
            # Encode using the model
            embeddings = self.encoder(texts)
            
            # Normalize for cosine similarity
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            
        return embeddings
    
    def add(self, texts, metadata=None):
        """새로운 문서 추가"""
        embeddings = self.encode(texts)
        
        if self.use_faiss:
            # Add to FAISS index
            self.index.add(embeddings.cpu().numpy())
        else:
            # Add to simple list
            self.vectors.extend(embeddings.cpu().numpy())
        
        # Store metadata
        if metadata is None:
            metadata = [{'text': t} for t in texts]
        self.metadata.extend(metadata)
    
    def search(self, query, k=5, threshold=None):
        """유사한 문서 검색"""
        query_embedding = self.encode(query)
        
        if self.use_faiss:
            # FAISS search
            distances, indices = self.index.search(query_embedding.cpu().numpy(), k)
            
            results = []
            for dist_row, idx_row in zip(distances, indices):
                row_results = []
                for dist, idx in zip(dist_row, idx_row):
                    if threshold is None or dist < threshold:
                        row_results.append({
                            'index': idx,
                            'distance': dist,
                            'similarity': 1 - dist / 2,  # Convert L2 to similarity
                            'metadata': self.metadata[idx] if idx < len(self.metadata) else None
                        })
                results.append(row_results)
            
            return results[0] if len(results) == 1 else results
        else:
            # Numpy search
            vectors_np = np.array(self.vectors)
            query_np = query_embedding.cpu().numpy()
            
            # Compute similarities
            similarities = np.dot(vectors_np, query_np.T).squeeze()
            
            # Get top-k
            top_k_idx = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_k_idx:
                if threshold is None or similarities[idx] > threshold:
                    results.append({
                        'index': idx,
                        'similarity': similarities[idx],
                        'metadata': self.metadata[idx]
                    })
            
            return results
    
    def batch_search(self, queries, k=5):
        """배치 검색"""
        all_results = []
        for query in queries:
            results = self.search(query, k)
            all_results.append(results)
        return all_results
    
    def remove(self, indices):
        """특정 인덱스 제거"""
        # Remove from metadata
        for idx in sorted(indices, reverse=True):
            if idx < len(self.metadata):
                del self.metadata[idx]
        
        # Note: FAISS doesn't support removal, need to rebuild
        if self.use_faiss:
            print("Warning: FAISS index removal requires rebuilding")

# 사용 예시
class SimpleEncoder(nn.Module):
    """간단한 텍스트 인코더"""
    def __init__(self, vocab_size=10000, embed_dim=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 8, batch_first=True),
            num_layers=2
        )
    
    def forward(self, texts):
        # Simplified: assume texts are tokenized tensors
        embeds = self.embedding(texts)
        encoded = self.transformer(embeds)
        return encoded.mean(dim=1)  # Pool

# 초기화 및 사용
encoder = SimpleEncoder()
retriever = DenseRetriever(encoder)

# 문서 추가
documents = [
    "Pick up the red ball",
    "Grasp the blue cube", 
    "Move the green cylinder"
]
retriever.add(documents)

# 검색
query = "grab the crimson sphere"
results = retriever.search(query, k=2)
print(f"Top results: {results}")
```

### 2. Sparse Retrieval (키워드 기반 + BM25)
```python
import math
from collections import Counter, defaultdict

class SparseRetriever:
    """BM25 기반 sparse retrieval"""
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1  # Term frequency saturation
        self.b = b    # Length normalization
        
        self.documents = []
        self.doc_lens = []
        self.doc_freqs = []  # Term frequencies per document
        self.df = defaultdict(int)  # Document frequency
        self.idf = {}  # Inverse document frequency
        self.avg_doc_len = 0
        self.N = 0  # Total number of documents
        
    def tokenize(self, text):
        """간단한 토큰화"""
        return text.lower().split()
    
    def add_documents(self, documents):
        """문서 추가 및 인덱싱"""
        for doc in documents:
            tokens = self.tokenize(doc)
            self.documents.append(doc)
            self.doc_lens.append(len(tokens))
            
            # Term frequency
            tf = Counter(tokens)
            self.doc_freqs.append(tf)
            
            # Document frequency
            for token in set(tokens):
                self.df[token] += 1
            
            self.N += 1
        
        # Calculate average document length
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0
        
        # Calculate IDF
        self._calculate_idf()
    
    def _calculate_idf(self):
        """IDF 계산"""
        for token, df in self.df.items():
            self.idf[token] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
    
    def score(self, query, doc_idx):
        """BM25 스코어 계산"""
        query_tokens = self.tokenize(query)
        doc_tf = self.doc_freqs[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        
        score = 0
        for token in query_tokens:
            if token not in self.idf:
                continue
            
            tf = doc_tf.get(token, 0)
            idf = self.idf[token]
            
            # BM25 formula
            numerator = idf * tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += numerator / denominator
        
        return score
    
    def search(self, query, k=5):
        """BM25 검색"""
        scores = []
        for idx in range(self.N):
            score = self.score(query, idx)
            scores.append((idx, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        results = []
        for idx, score in scores[:k]:
            results.append({
                'index': idx,
                'score': score,
                'document': self.documents[idx]
            })
        
        return results

# 사용 예시
sparse_retriever = SparseRetriever()
sparse_retriever.add_documents([
    "Pick up the red ball from the table",
    "Place the blue cube in the box",
    "Move the robot arm to the left"
])

results = sparse_retriever.search("get red ball", k=2)
for r in results:
    print(f"Score: {r['score']:.3f}, Doc: {r['document']}")
```

### 3. Hybrid Retrieval (Dense + Sparse)
```python
class HybridRetriever:
    """Dense와 Sparse retrieval 결합"""
    def __init__(self, dense_retriever, sparse_retriever, alpha=0.5):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.alpha = alpha  # Weight for dense scores
        
    def add(self, documents, metadata=None):
        """문서 추가"""
        self.dense_retriever.add(documents, metadata)
        self.sparse_retriever.add_documents(documents)
    
    def search(self, query, k=5):
        """하이브리드 검색"""
        # Get results from both retrievers
        dense_results = self.dense_retriever.search(query, k=k*2)
        sparse_results = self.sparse_retriever.search(query, k=k*2)
        
        # Combine scores
        combined_scores = {}
        
        # Add dense scores
        for i, result in enumerate(dense_results):
            idx = result['index']
            combined_scores[idx] = self.alpha * result['similarity']
        
        # Add sparse scores (normalized)
        max_sparse_score = max([r['score'] for r in sparse_results]) if sparse_results else 1
        for result in sparse_results:
            idx = result['index']
            normalized_score = result['score'] / max_sparse_score if max_sparse_score > 0 else 0
            if idx in combined_scores:
                combined_scores[idx] += (1 - self.alpha) * normalized_score
            else:
                combined_scores[idx] = (1 - self.alpha) * normalized_score
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format results
        final_results = []
        for idx, score in sorted_results[:k]:
            final_results.append({
                'index': idx,
                'score': score,
                'document': self.sparse_retriever.documents[idx] if idx < len(self.sparse_retriever.documents) else None,
                'metadata': self.dense_retriever.metadata[idx] if idx < len(self.dense_retriever.metadata) else None
            })
        
        return final_results
```

### 4. Vector Database Implementation
```python
class VectorDatabase:
    """완전한 벡터 데이터베이스 구현"""
    def __init__(self, dimension=768, metric='cosine'):
        self.dimension = dimension
        self.metric = metric
        
        # Storage
        self.vectors = []
        self.metadata = []
        self.ids = []
        self.id_counter = 0
        
        # Indexes for different metrics
        self.indices = {}
        
        # Cache for frequently accessed items
        self.cache = {}
        self.cache_size = 100
        
    def add(self, vectors, metadata=None, ids=None):
        """벡터 추가"""
        if isinstance(vectors, torch.Tensor):
            vectors = vectors.cpu().numpy()
        
        batch_size = vectors.shape[0]
        
        # Generate IDs if not provided
        if ids is None:
            ids = list(range(self.id_counter, self.id_counter + batch_size))
            self.id_counter += batch_size
        
        # Store
        self.vectors.extend(vectors)
        self.ids.extend(ids)
        
        if metadata is None:
            metadata = [{}] * batch_size
        self.metadata.extend(metadata)
        
        # Update indices
        self._update_indices()
        
        return ids
    
    def _update_indices(self):
        """인덱스 업데이트"""
        if len(self.vectors) > 0:
            vectors_np = np.array(self.vectors)
            
            # Build different index types
            if self.metric == 'cosine':
                # Normalize for cosine similarity
                norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
                self.indices['normalized'] = vectors_np / (norms + 1e-10)
            elif self.metric == 'euclidean':
                self.indices['raw'] = vectors_np
    
    def search(self, query_vector, k=5, filter_metadata=None):
        """벡터 검색"""
        if isinstance(query_vector, torch.Tensor):
            query_vector = query_vector.cpu().numpy()
        
        # Check cache
        cache_key = query_vector.tobytes()
        if cache_key in self.cache:
            return self.cache[cache_key][:k]
        
        # Compute distances
        if self.metric == 'cosine':
            vectors = self.indices.get('normalized', np.array(self.vectors))
            query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
            similarities = np.dot(vectors, query_norm)
            distances = 1 - similarities
        elif self.metric == 'euclidean':
            vectors = self.indices.get('raw', np.array(self.vectors))
            distances = np.linalg.norm(vectors - query_vector, axis=1)
        
        # Apply metadata filter if provided
        if filter_metadata:
            valid_indices = []
            for i, meta in enumerate(self.metadata):
                if all(meta.get(k) == v for k, v in filter_metadata.items()):
                    valid_indices.append(i)
            
            if valid_indices:
                filtered_distances = distances[valid_indices]
                sorted_indices = np.argsort(filtered_distances)[:k]
                final_indices = [valid_indices[i] for i in sorted_indices]
            else:
                return []
        else:
            final_indices = np.argsort(distances)[:k]
        
        # Prepare results
        results = []
        for idx in final_indices:
            results.append({
                'id': self.ids[idx],
                'distance': distances[idx],
                'vector': self.vectors[idx],
                'metadata': self.metadata[idx]
            })
        
        # Update cache
        self.cache[cache_key] = results
        if len(self.cache) > self.cache_size:
            # Remove oldest cache entry
            self.cache.pop(list(self.cache.keys())[0])
        
        return results[:k]
    
    def delete(self, ids):
        """벡터 삭제"""
        indices_to_delete = [self.ids.index(id) for id in ids if id in self.ids]
        
        for idx in sorted(indices_to_delete, reverse=True):
            del self.vectors[idx]
            del self.metadata[idx]
            del self.ids[idx]
        
        # Rebuild indices
        self._update_indices()
        
        # Clear cache
        self.cache.clear()
    
    def update(self, id, vector=None, metadata=None):
        """벡터 또는 메타데이터 업데이트"""
        if id not in self.ids:
            raise ValueError(f"ID {id} not found")
        
        idx = self.ids.index(id)
        
        if vector is not None:
            if isinstance(vector, torch.Tensor):
                vector = vector.cpu().numpy()
            self.vectors[idx] = vector
            self._update_indices()
        
        if metadata is not None:
            self.metadata[idx].update(metadata)
        
        # Clear cache
        self.cache.clear()
    
    def save(self, path):
        """데이터베이스 저장"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'vectors': self.vectors,
                'metadata': self.metadata,
                'ids': self.ids,
                'id_counter': self.id_counter,
                'dimension': self.dimension,
                'metric': self.metric
            }, f)
    
    def load(self, path):
        """데이터베이스 로드"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectors = data['vectors']
            self.metadata = data['metadata']
            self.ids = data['ids']
            self.id_counter = data['id_counter']
            self.dimension = data['dimension']
            self.metric = data['metric']
            self._update_indices()
```

---

## 🤖 VLA에서의 활용

### 1. RAG-VLA System
```python
class RAG_VLA(nn.Module):
    """Retrieval-Augmented VLA 시스템"""
    def __init__(self, vision_encoder, language_encoder, action_decoder, vector_db):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_encoder = language_encoder
        self.action_decoder = action_decoder
        self.vector_db = vector_db
        
        # Fusion network for retrieved context
        self.context_fusion = nn.Sequential(
            nn.Linear(768 * 3, 1024),  # current + retrieved + instruction
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 768)
        )
        
        # Relevance scoring
        self.relevance_scorer = nn.Sequential(
            nn.Linear(768 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image, instruction, store_experience=True):
        # Encode current observation
        vision_features = self.vision_encoder(image)
        instruction_features = self.language_encoder(instruction)
        
        # Create query for retrieval
        query = torch.cat([vision_features, instruction_features], dim=-1)
        
        # Retrieve similar past experiences
        retrieved_results = self.vector_db.search(
            query_vector=query,
            k=5,
            filter_metadata={'success': True}  # Only retrieve successful experiences
        )
        
        # Process retrieved experiences
        if retrieved_results:
            retrieved_vectors = torch.stack([
                torch.tensor(r['vector']) for r in retrieved_results
            ])
            
            # Score relevance of each retrieved experience
            relevance_scores = []
            for ret_vec in retrieved_vectors:
                score = self.relevance_scorer(
                    torch.cat([query, ret_vec.unsqueeze(0)], dim=-1)
                )
                relevance_scores.append(score)
            
            relevance_scores = torch.stack(relevance_scores)
            
            # Weighted aggregation of retrieved experiences
            weighted_context = (retrieved_vectors * relevance_scores).sum(dim=0)
        else:
            weighted_context = torch.zeros_like(query)
        
        # Fuse current observation with retrieved context
        fused_features = self.context_fusion(
            torch.cat([vision_features, instruction_features, weighted_context], dim=-1)
        )
        
        # Generate action
        action = self.action_decoder(fused_features)
        
        # Store current experience if specified
        if store_experience:
            experience_vector = torch.cat([vision_features, instruction_features, action], dim=-1)
            self.vector_db.add(
                vectors=experience_vector,
                metadata={
                    'timestamp': torch.tensor(0),  # Would be actual timestamp
                    'success': None  # To be updated later based on outcome
                }
            )
        
        return action, {
            'retrieved_count': len(retrieved_results),
            'relevance_scores': relevance_scores if retrieved_results else None
        }
```

### 2. Skill Retrieval System
```python
class SkillLibrary:
    """재사용 가능한 스킬 라이브러리"""
    def __init__(self, encoder_model):
        self.encoder = encoder_model
        self.vector_db = VectorDatabase(dimension=768)
        self.skills = {}
        
    def add_skill(self, name, description, demonstration_trajectory, prerequisites=None):
        """새로운 스킬 추가"""
        # Encode skill description
        skill_embedding = self.encoder(description)
        
        # Store in vector database
        skill_id = self.vector_db.add(
            vectors=skill_embedding,
            metadata={
                'name': name,
                'description': description,
                'prerequisites': prerequisites or []
            }
        )[0]
        
        # Store skill details
        self.skills[skill_id] = {
            'name': name,
            'trajectory': demonstration_trajectory,
            'embedding': skill_embedding,
            'usage_count': 0,
            'success_rate': 0.0
        }
        
        return skill_id
    
    def retrieve_skill(self, task_description, context=None):
        """태스크에 적합한 스킬 검색"""
        # Encode task
        task_embedding = self.encoder(task_description)
        
        # Search for similar skills
        results = self.vector_db.search(task_embedding, k=3)
        
        if not results:
            return None
        
        # Check prerequisites if context provided
        if context:
            valid_skills = []
            for result in results:
                skill = self.skills[result['id']]
                prerequisites = result['metadata'].get('prerequisites', [])
                
                # Check if all prerequisites are met
                if all(prereq in context for prereq in prerequisites):
                    valid_skills.append(result)
            
            if valid_skills:
                results = valid_skills
        
        # Return best matching skill
        best_result = results[0]
        skill = self.skills[best_result['id']]
        
        # Update usage statistics
        skill['usage_count'] += 1
        
        return {
            'id': best_result['id'],
            'name': skill['name'],
            'trajectory': skill['trajectory'],
            'similarity': 1 - best_result['distance']
        }
    
    def update_skill_performance(self, skill_id, success):
        """스킬 성능 업데이트"""
        if skill_id in self.skills:
            skill = self.skills[skill_id]
            # Update success rate with moving average
            alpha = 0.1
            skill['success_rate'] = (1 - alpha) * skill['success_rate'] + alpha * (1.0 if success else 0.0)
    
    def get_skill_hierarchy(self):
        """스킬 계층 구조 생성"""
        hierarchy = {}
        
        for skill_id, skill in self.skills.items():
            metadata = self.vector_db.metadata[self.vector_db.ids.index(skill_id)]
            prerequisites = metadata.get('prerequisites', [])
            
            if not prerequisites:
                # Base skill
                hierarchy[skill['name']] = {'level': 0, 'children': []}
            else:
                # Composite skill
                hierarchy[skill['name']] = {
                    'level': len(prerequisites),
                    'prerequisites': prerequisites,
                    'children': []
                }
        
        return hierarchy
```

### 3. Contextual Retrieval for VLA
```python
class ContextualRetriever:
    """컨텍스트 aware 검색 시스템"""
    def __init__(self, embedding_dim=768):
        self.embedding_dim = embedding_dim
        
        # Multiple vector stores for different modalities
        self.vision_db = VectorDatabase(dimension=embedding_dim, metric='cosine')
        self.language_db = VectorDatabase(dimension=embedding_dim, metric='cosine')
        self.action_db = VectorDatabase(dimension=embedding_dim, metric='euclidean')
        
        # Cross-modal retrieval
        self.cross_modal_proj = nn.Linear(embedding_dim * 2, embedding_dim)
        
    def add_experience(self, vision_embed, language_embed, action_embed, outcome):
        """멀티모달 경험 추가"""
        timestamp = torch.tensor([time.time()])
        
        # Add to respective databases
        vision_id = self.vision_db.add(
            vision_embed,
            metadata={'timestamp': timestamp, 'outcome': outcome}
        )[0]
        
        language_id = self.language_db.add(
            language_embed,
            metadata={'timestamp': timestamp, 'outcome': outcome}
        )[0]
        
        action_id = self.action_db.add(
            action_embed,
            metadata={'timestamp': timestamp, 'outcome': outcome}
        )[0]
        
        # Link the IDs
        return {
            'vision_id': vision_id,
            'language_id': language_id,
            'action_id': action_id
        }
    
    def retrieve_multimodal(self, query_vision=None, query_language=None, k=5):
        """멀티모달 검색"""
        results = []
        
        if query_vision is not None and query_language is not None:
            # Cross-modal retrieval
            cross_modal_query = self.cross_modal_proj(
                torch.cat([query_vision, query_language], dim=-1)
            )
            
            # Search in both databases
            vision_results = self.vision_db.search(cross_modal_query, k=k)
            language_results = self.language_db.search(cross_modal_query, k=k)
            
            # Merge results
            seen_timestamps = set()
            for v_result, l_result in zip(vision_results, language_results):
                v_time = v_result['metadata']['timestamp'].item()
                l_time = l_result['metadata']['timestamp'].item()
                
                if v_time not in seen_timestamps:
                    results.append(v_result)
                    seen_timestamps.add(v_time)
                
                if l_time not in seen_timestamps:
                    results.append(l_result)
                    seen_timestamps.add(l_time)
        
        elif query_vision is not None:
            results = self.vision_db.search(query_vision, k=k)
        
        elif query_language is not None:
            results = self.language_db.search(query_language, k=k)
        
        return results[:k]
    
    def adaptive_retrieval(self, query, context_history, k=5):
        """적응적 검색 (컨텍스트 고려)"""
        # Weight recent context more heavily
        context_weights = torch.exp(-torch.arange(len(context_history), dtype=torch.float32) * 0.1)
        weighted_context = torch.sum(
            torch.stack(context_history) * context_weights.unsqueeze(-1),
            dim=0
        ) / context_weights.sum()
        
        # Combine query with weighted context
        adapted_query = 0.7 * query + 0.3 * weighted_context
        
        # Retrieve with adapted query
        results = self.vision_db.search(adapted_query, k=k)
        
        # Re-rank based on temporal proximity
        current_time = time.time()
        for result in results:
            time_diff = current_time - result['metadata']['timestamp'].item()
            # Boost recent experiences
            recency_weight = math.exp(-time_diff / 3600)  # Decay over hours
            result['final_score'] = result['distance'] * (1 + 0.2 * recency_weight)
        
        # Sort by final score
        results.sort(key=lambda x: x['final_score'])
        
        return results
```

---

## 🔬 핵심 개념 정리

### 1. Retrieval Metrics
```python
def evaluate_retrieval(retriever, test_queries, ground_truth):
    """검색 시스템 평가 메트릭"""
    metrics = {
        'precision_at_k': [],
        'recall_at_k': [],
        'mrr': [],  # Mean Reciprocal Rank
        'ndcg': []  # Normalized Discounted Cumulative Gain
    }
    
    for query, relevant_docs in zip(test_queries, ground_truth):
        results = retriever.search(query, k=10)
        retrieved_ids = [r['index'] for r in results]
        
        # Precision@K
        k = 5
        relevant_in_top_k = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_docs)
        precision_k = relevant_in_top_k / k
        metrics['precision_at_k'].append(precision_k)
        
        # Recall@K
        recall_k = relevant_in_top_k / len(relevant_docs) if relevant_docs else 0
        metrics['recall_at_k'].append(recall_k)
        
        # MRR
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_docs:
                metrics['mrr'].append(1 / (i + 1))
                break
        else:
            metrics['mrr'].append(0)
        
        # NDCG
        dcg = sum(
            1 / math.log2(i + 2) for i, doc_id in enumerate(retrieved_ids)
            if doc_id in relevant_docs
        )
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant_docs), len(retrieved_ids))))
        ndcg = dcg / idcg if idcg > 0 else 0
        metrics['ndcg'].append(ndcg)
    
    # Average metrics
    return {
        metric: sum(values) / len(values) if values else 0
        for metric, values in metrics.items()
    }
```

### 2. Index Optimization
```python
def optimize_index(vector_db, data_characteristics):
    """인덱스 최적화 전략 선택"""
    num_vectors = len(vector_db.vectors)
    dimension = vector_db.dimension
    
    if num_vectors < 1000:
        # Small dataset: brute force is fine
        return "flat"
    elif num_vectors < 10000:
        # Medium dataset: IVF for balance
        return "ivf"
    elif num_vectors < 100000:
        # Large dataset: HNSW for speed
        return "hnsw"
    else:
        # Very large: need approximate methods
        return "lsh"  # Locality Sensitive Hashing
```

---

## 🛠️ 실습 코드

### 완전한 RAG-VLA 시스템 구현
```python
import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Optional, Tuple

class CompleteRAGVLA:
    """Production-ready RAG-VLA 시스템"""
    def __init__(self, 
                 vision_model_name="resnet50",
                 language_model_name="bert-base",
                 vector_db_path=None):
        
        # Initialize encoders
        self.vision_encoder = self._load_vision_encoder(vision_model_name)
        self.language_encoder = self._load_language_encoder(language_model_name)
        
        # Initialize vector databases
        self.experience_db = VectorDatabase(dimension=768, metric='cosine')
        self.skill_db = VectorDatabase(dimension=768, metric='cosine')
        
        # Initialize retrievers
        self.dense_retriever = DenseRetriever(
            encoder_model=self.language_encoder,
            vector_dim=768
        )
        
        # Action generation network
        self.action_generator = nn.Sequential(
            nn.Linear(768 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 7)  # 7-DOF action
        )
        
        # Load existing database if provided
        if vector_db_path:
            self.load_database(vector_db_path)
        
        # Statistics tracking
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'avg_retrieval_time': 0,
            'success_rate': 0
        }
    
    def _load_vision_encoder(self, model_name):
        """비전 인코더 로드"""
        import torchvision.models as models
        
        if model_name == "resnet50":
            model = models.resnet50(pretrained=True)
            # Remove classification head
            model = nn.Sequential(*list(model.children())[:-1])
            # Add projection to common dimension
            return nn.Sequential(
                model,
                nn.Flatten(),
                nn.Linear(2048, 768)
            )
        else:
            raise ValueError(f"Unknown vision model: {model_name}")
    
    def _load_language_encoder(self, model_name):
        """언어 인코더 로드"""
        # Simplified encoder
        return nn.Sequential(
            nn.Embedding(10000, 768),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(768, 8, batch_first=True),
                num_layers=2
            )
        )
    
    def process_task(self, image, instruction, context=None):
        """태스크 처리 파이프라인"""
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        # Encode inputs
        with torch.no_grad():
            vision_features = self.vision_encoder(image)
            # Simplified: assume instruction is already tokenized
            language_features = self.language_encoder(instruction).mean(dim=1)
        
        # Retrieve relevant experiences
        query = torch.cat([vision_features, language_features], dim=-1)
        retrieved_experiences = self.experience_db.search(query, k=5)
        
        # Retrieve relevant skills
        skill_query = language_features
        retrieved_skills = self.skill_db.search(skill_query, k=3)
        
        # Combine retrieved information
        if retrieved_experiences:
            experience_context = torch.mean(
                torch.stack([torch.tensor(e['vector']) for e in retrieved_experiences]),
                dim=0
            )
        else:
            experience_context = torch.zeros(768)
        
        if retrieved_skills:
            skill_context = torch.mean(
                torch.stack([torch.tensor(s['vector']) for s in retrieved_skills]),
                dim=0
            )
        else:
            skill_context = torch.zeros(768)
        
        # Generate action
        action_input = torch.cat([
            vision_features.squeeze(),
            language_features.squeeze(),
            experience_context
        ])
        
        action = self.action_generator(action_input)
        
        # Update statistics
        retrieval_time = time.time() - start_time
        self.stats['avg_retrieval_time'] = (
            self.stats['avg_retrieval_time'] * (self.stats['total_queries'] - 1) +
            retrieval_time
        ) / self.stats['total_queries']
        
        return {
            'action': action,
            'retrieved_experiences': len(retrieved_experiences),
            'retrieved_skills': len(retrieved_skills),
            'retrieval_time': retrieval_time,
            'confidence': self._estimate_confidence(retrieved_experiences, retrieved_skills)
        }
    
    def _estimate_confidence(self, experiences, skills):
        """신뢰도 추정"""
        if not experiences and not skills:
            return 0.3
        
        exp_confidence = min(len(experiences) / 5, 1.0) if experiences else 0
        skill_confidence = min(len(skills) / 3, 1.0) if skills else 0
        
        # Average similarity of top results
        if experiences:
            exp_similarity = 1 - experiences[0]['distance']
        else:
            exp_similarity = 0
        
        return 0.4 * exp_confidence + 0.3 * skill_confidence + 0.3 * exp_similarity
    
    def add_experience(self, image, instruction, action, outcome):
        """새로운 경험 추가"""
        with torch.no_grad():
            vision_features = self.vision_encoder(image)
            language_features = self.language_encoder(instruction).mean(dim=1)
        
        experience_vector = torch.cat([
            vision_features.squeeze(),
            language_features.squeeze(),
            action.squeeze()
        ])
        
        self.experience_db.add(
            vectors=experience_vector.unsqueeze(0),
            metadata={
                'timestamp': time.time(),
                'outcome': outcome,
                'success': outcome > 0.5
            }
        )
    
    def add_skill(self, name, description, demonstration):
        """새로운 스킬 추가"""
        # Encode skill description
        skill_vector = self.language_encoder(description).mean(dim=1)
        
        self.skill_db.add(
            vectors=skill_vector,
            metadata={
                'name': name,
                'description': description,
                'demonstration_length': len(demonstration)
            }
        )
    
    def optimize_retrieval(self):
        """검색 최적화"""
        # Remove old unsuccessful experiences
        current_time = time.time()
        to_remove = []
        
        for i, metadata in enumerate(self.experience_db.metadata):
            age = current_time - metadata['timestamp']
            if age > 3600 * 24 and not metadata.get('success', False):
                to_remove.append(self.experience_db.ids[i])
        
        if to_remove:
            self.experience_db.delete(to_remove)
            print(f"Removed {len(to_remove)} old unsuccessful experiences")
        
        # Rebuild indices
        self.experience_db._update_indices()
        self.skill_db._update_indices()
    
    def get_statistics(self):
        """시스템 통계 반환"""
        return {
            **self.stats,
            'experience_count': len(self.experience_db.vectors),
            'skill_count': len(self.skill_db.vectors),
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['total_queries'], 1)
        }
    
    def save_database(self, path):
        """데이터베이스 저장"""
        self.experience_db.save(f"{path}_experiences.pkl")
        self.skill_db.save(f"{path}_skills.pkl")
    
    def load_database(self, path):
        """데이터베이스 로드"""
        self.experience_db.load(f"{path}_experiences.pkl")
        self.skill_db.load(f"{path}_skills.pkl")

# 사용 예시
def demo_rag_vla():
    # Initialize system
    rag_vla = CompleteRAGVLA()
    
    # Simulate robot task
    image = torch.randn(1, 3, 224, 224)
    instruction = torch.randint(0, 10000, (1, 20))  # Tokenized instruction
    
    # Process task
    result = rag_vla.process_task(image, instruction)
    print(f"Generated action: {result['action']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Retrieved {result['retrieved_experiences']} experiences")
    print(f"Retrieval time: {result['retrieval_time']:.3f}s")
    
    # Add experience
    outcome = 0.8  # Successful
    rag_vla.add_experience(image, instruction, result['action'], outcome)
    
    # Add skill
    rag_vla.add_skill(
        name="pick_and_place",
        description=torch.randint(0, 10000, (1, 15)),
        demonstration=[torch.randn(7) for _ in range(10)]
    )
    
    # Get statistics
    stats = rag_vla.get_statistics()
    print(f"\nSystem Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Optimize retrieval
    rag_vla.optimize_retrieval()
    
    # Save database
    rag_vla.save_database("rag_vla_checkpoint")

if __name__ == "__main__":
    demo_rag_vla()
```

---

## 📈 다음 단계

### 1. 고급 검색 기법
- **Learned Indices**: 학습 가능한 인덱스 구조
- **Neural Information Retrieval**: 엔드투엔드 학습
- **Graph-based Retrieval**: 관계 정보 활용

### 2. VLA 특화 개선
- **Temporal Retrieval**: 시간적 패턴 검색
- **Multi-Robot Sharing**: 로봇 간 경험 공유
- **Active Learning**: 검색 결과로 학습 우선순위 결정

### 3. 확장성
- **Distributed Retrieval**: 분산 검색 시스템
- **Streaming Updates**: 실시간 인덱스 업데이트
- **Compression**: 효율적 저장 및 전송

---

## 💡 핵심 포인트

### ✅ 기억해야 할 것들
1. **Dense vs Sparse**: 각각의 장단점 이해
2. **Hybrid Approach**: 두 방법의 장점 결합
3. **Vector Database**: 효율적인 저장과 검색
4. **RAG Pattern**: 검색 증강 생성의 강력함

### ⚠️ 주의사항
1. **검색 지연**: 실시간 시스템에서 중요
2. **메모리 사용**: 대규모 인덱스 관리
3. **품질 vs 속도**: 트레이드오프 고려

### 🎯 VLA 적용 시
1. **Experience Replay**: 과거 경험 효율적 활용
2. **Skill Transfer**: 학습된 스킬 재사용
3. **Contextual Retrieval**: 상황 맞춤 검색

---

**다음 문서**: `05_vision_encoders.md` - 로봇 비전을 위한 인코더