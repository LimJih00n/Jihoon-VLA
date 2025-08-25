# 🧠 Memory & Context: 로봇의 기억 시스템

**목표**: Working Memory, Long-term Memory, Context Management의 이해 및 VLA 적용  
**시간**: 2-3시간  
**전제조건**: 01_neural_networks_basics.md, 02_attention_mechanism.md  

---

## 🎯 개발자를 위한 직관적 이해

### 인간의 기억 vs AI의 기억
```python
human_memory = {
    "sensory": "1-2초 유지 (시각, 청각 입력)",
    "working": "7±2개 항목, 20-30초 (전화번호 외우기)",
    "long_term": "무제한 용량, 영구 저장 (자전거 타기)"
}

ai_memory = {
    "attention": "현재 context window (sensory)",
    "cache": "KV cache, hidden states (working)",
    "retrieval": "Vector DB, external storage (long-term)"
}
```

### VLA에서 왜 중요한가?
- **Task Continuity**: 복잡한 작업의 단계별 기억
- **Context Awareness**: 환경과 상황 인식
- **Learning from Experience**: 과거 경험 활용

---

## 🏗️ 기본 구조 및 구현

### 1. Working Memory Implementation
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

class WorkingMemory(nn.Module):
    """단기 작업 메모리 구현"""
    def __init__(self, memory_size=7, feature_dim=768):
        super().__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        
        # Memory slots
        self.memory_slots = nn.Parameter(torch.randn(memory_size, feature_dim))
        
        # Memory controller
        self.write_head = nn.Linear(feature_dim, feature_dim)
        self.read_head = nn.Linear(feature_dim, feature_dim)
        self.erase_head = nn.Linear(feature_dim, feature_dim)
        
        # Attention mechanism for memory access
        self.memory_attention = nn.MultiheadAttention(
            feature_dim, num_heads=8, batch_first=True
        )
        
    def write(self, input_features, memory_state):
        """메모리에 정보 쓰기"""
        batch_size = input_features.shape[0]
        
        # Compute write weights (attention)
        write_key = self.write_head(input_features)
        write_weights = F.softmax(
            torch.matmul(write_key, memory_state.transpose(-2, -1)) / self.feature_dim**0.5,
            dim=-1
        )
        
        # Erase vector
        erase_vector = torch.sigmoid(self.erase_head(input_features))
        
        # Update memory
        # 1. Erase
        memory_state = memory_state * (1 - write_weights.unsqueeze(-1) * erase_vector.unsqueeze(1))
        # 2. Write
        memory_state = memory_state + write_weights.unsqueeze(-1) * input_features.unsqueeze(1)
        
        return memory_state
    
    def read(self, query, memory_state):
        """메모리에서 정보 읽기"""
        # Use attention to read from memory
        read_key = self.read_head(query)
        
        # Attention-based read
        output, attention_weights = self.memory_attention(
            query=read_key.unsqueeze(1),
            key=memory_state,
            value=memory_state
        )
        
        return output.squeeze(1), attention_weights
    
    def forward(self, input_features, memory_state=None):
        batch_size = input_features.shape[0]
        
        # Initialize memory if not provided
        if memory_state is None:
            memory_state = self.memory_slots.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Write to memory
        memory_state = self.write(input_features, memory_state)
        
        # Read from memory
        output, read_weights = self.read(input_features, memory_state)
        
        return output, memory_state, read_weights

# 사용 예시
working_mem = WorkingMemory(memory_size=7, feature_dim=768)
input_features = torch.randn(2, 768)  # batch_size=2

output, updated_memory, attention = working_mem(input_features)
print(f"Output: {output.shape}")  # [2, 768]
print(f"Memory state: {updated_memory.shape}")  # [2, 7, 768]
```

### 2. Long-term Memory with External Storage
```python
class LongTermMemory(nn.Module):
    """외부 저장소를 활용한 장기 메모리"""
    def __init__(self, memory_bank_size=1000, feature_dim=768, top_k=5):
        super().__init__()
        self.memory_bank_size = memory_bank_size
        self.feature_dim = feature_dim
        self.top_k = top_k
        
        # Memory bank (can be saved/loaded)
        self.memory_bank = nn.Parameter(
            torch.randn(memory_bank_size, feature_dim),
            requires_grad=False
        )
        self.memory_keys = nn.Parameter(
            torch.randn(memory_bank_size, feature_dim),
            requires_grad=False
        )
        self.memory_ages = torch.zeros(memory_bank_size)  # Track memory age
        
        # Query projection
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        # Memory write controller
        self.write_gate = nn.Linear(feature_dim * 2, 1)
        
    def retrieve(self, query):
        """관련 메모리 검색"""
        # Project query
        query_vector = self.query_proj(query)
        
        # Compute similarities
        similarities = F.cosine_similarity(
            query_vector.unsqueeze(1),
            self.memory_keys.unsqueeze(0),
            dim=-1
        )
        
        # Get top-k memories
        top_k_values, top_k_indices = torch.topk(similarities, self.top_k, dim=-1)
        
        # Retrieve memories
        retrieved_memories = self.memory_bank[top_k_indices]
        
        # Weight by similarity
        weights = F.softmax(top_k_values, dim=-1)
        weighted_memory = (retrieved_memories * weights.unsqueeze(-1)).sum(dim=1)
        
        return weighted_memory, top_k_indices, weights
    
    def store(self, new_memory, importance_score=None):
        """새로운 메모리 저장"""
        batch_size = new_memory.shape[0]
        
        # Compute memory key
        memory_key = self.value_proj(new_memory)
        
        # Find least important/oldest memories to replace
        if importance_score is None:
            # Use age as importance (older = less important)
            _, replace_indices = torch.topk(-self.memory_ages, batch_size)
        else:
            # Use provided importance scores
            _, replace_indices = torch.topk(-importance_score, batch_size)
        
        # Update memory bank
        self.memory_bank.data[replace_indices] = new_memory.detach()
        self.memory_keys.data[replace_indices] = memory_key.detach()
        self.memory_ages[replace_indices] = 0
        
        # Age all memories
        self.memory_ages += 1
        
    def consolidate(self, threshold=0.9):
        """메모리 통합 (유사한 메모리 병합)"""
        # Compute pairwise similarities
        similarities = F.cosine_similarity(
            self.memory_keys.unsqueeze(1),
            self.memory_keys.unsqueeze(0),
            dim=-1
        )
        
        # Find similar memories (above threshold)
        similar_pairs = (similarities > threshold).nonzero()
        similar_pairs = similar_pairs[similar_pairs[:, 0] < similar_pairs[:, 1]]
        
        # Merge similar memories
        for i, j in similar_pairs:
            # Average the memories
            self.memory_bank.data[i] = (self.memory_bank[i] + self.memory_bank[j]) / 2
            self.memory_keys.data[i] = (self.memory_keys[i] + self.memory_keys[j]) / 2
            # Mark j for removal
            self.memory_ages[j] = float('inf')
        
    def forward(self, query, store_new=False):
        # Retrieve relevant memories
        retrieved, indices, weights = self.retrieve(query)
        
        # Combine with query
        combined = torch.cat([query, retrieved], dim=-1)
        
        # Decide whether to store
        if store_new:
            write_prob = torch.sigmoid(self.write_gate(combined))
            if write_prob > 0.5:
                self.store(query)
        
        return retrieved, weights

# 사용 예시
ltm = LongTermMemory(memory_bank_size=1000, feature_dim=768)

# Store some memories
experiences = torch.randn(10, 768)
for exp in experiences:
    ltm.store(exp.unsqueeze(0))

# Retrieve relevant memory
query = torch.randn(1, 768)
retrieved_memory, weights = ltm(query)
print(f"Retrieved memory: {retrieved_memory.shape}")  # [1, 768]
```

### 3. Episodic Memory for Robotics
```python
class EpisodicMemory(nn.Module):
    """로봇 작업을 위한 에피소드 메모리"""
    def __init__(self, episode_length=100, feature_dim=768):
        super().__init__()
        self.episode_length = episode_length
        self.feature_dim = feature_dim
        
        # Episode storage
        self.episodes = []
        self.current_episode = []
        
        # Episode encoder
        self.episode_encoder = nn.LSTM(
            feature_dim, feature_dim // 2, 
            bidirectional=True, batch_first=True
        )
        
        # Episode similarity computation
        self.similarity_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def add_timestep(self, observation, action, reward=None):
        """현재 에피소드에 timestep 추가"""
        timestep = {
            'observation': observation,
            'action': action,
            'reward': reward,
            'timestamp': len(self.current_episode)
        }
        self.current_episode.append(timestep)
        
        # Check if episode is complete
        if len(self.current_episode) >= self.episode_length:
            self.finish_episode()
    
    def finish_episode(self, success=None):
        """에피소드 완료 및 저장"""
        if len(self.current_episode) > 0:
            episode_data = {
                'steps': self.current_episode,
                'length': len(self.current_episode),
                'success': success,
                'encoded': self._encode_episode(self.current_episode)
            }
            self.episodes.append(episode_data)
            self.current_episode = []
    
    def _encode_episode(self, episode):
        """에피소드를 고정 크기 벡터로 인코딩"""
        # Stack observations
        observations = torch.stack([step['observation'] for step in episode])
        
        # Encode with LSTM
        encoded, (hidden, cell) = self.episode_encoder(observations.unsqueeze(0))
        
        # Use final hidden state as episode representation
        episode_repr = torch.cat([hidden[0], hidden[1]], dim=-1)
        
        return episode_repr.squeeze(0)
    
    def retrieve_similar_episodes(self, current_context, k=3):
        """현재 상황과 유사한 과거 에피소드 검색"""
        if len(self.episodes) == 0:
            return []
        
        # Encode current context
        if len(self.current_episode) > 0:
            current_encoded = self._encode_episode(self.current_episode)
        else:
            current_encoded = current_context
        
        # Compute similarities with all stored episodes
        similarities = []
        for episode in self.episodes:
            ep_encoded = episode['encoded']
            sim_input = torch.cat([current_encoded, ep_encoded], dim=-1)
            similarity = self.similarity_net(sim_input)
            similarities.append(similarity.item())
        
        # Get top-k similar episodes
        top_k_indices = sorted(range(len(similarities)), 
                              key=lambda i: similarities[i], 
                              reverse=True)[:k]
        
        return [self.episodes[i] for i in top_k_indices]
    
    def get_action_from_similar_episodes(self, current_obs, k=3):
        """유사한 에피소드에서 행동 추천"""
        similar_episodes = self.retrieve_similar_episodes(current_obs, k)
        
        if not similar_episodes:
            return None
        
        # Find similar states in retrieved episodes
        recommended_actions = []
        for episode in similar_episodes:
            # Find most similar observation in episode
            best_match_idx = 0
            best_similarity = -float('inf')
            
            for i, step in enumerate(episode['steps']):
                similarity = F.cosine_similarity(
                    current_obs.unsqueeze(0),
                    step['observation'].unsqueeze(0)
                ).item()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_idx = i
            
            # Get action from next step
            if best_match_idx < len(episode['steps']) - 1:
                recommended_actions.append(
                    episode['steps'][best_match_idx + 1]['action']
                )
        
        if recommended_actions:
            # Average recommended actions
            return torch.stack(recommended_actions).mean(dim=0)
        
        return None
```

### 4. Hierarchical Context Management
```python
class HierarchicalContext(nn.Module):
    """계층적 컨텍스트 관리 (L1, L2, L3 메모리)"""
    def __init__(self, feature_dim=768):
        super().__init__()
        
        # L1: Immediate context (1-5 timesteps)
        self.l1_buffer = deque(maxlen=5)
        self.l1_encoder = nn.GRU(feature_dim, feature_dim, batch_first=True)
        
        # L2: Working context (5-50 timesteps)
        self.l2_buffer = deque(maxlen=50)
        self.l2_encoder = nn.LSTM(feature_dim, feature_dim, batch_first=True)
        
        # L3: Long-term context (50+ timesteps)
        self.l3_memory = LongTermMemory(memory_bank_size=500, feature_dim=feature_dim)
        
        # Context fusion
        self.context_fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # Attention weights for different levels
        self.level_attention = nn.Parameter(torch.ones(3) / 3)
        
    def update(self, new_observation):
        """새로운 관찰로 모든 레벨 업데이트"""
        # Update L1 (immediate)
        self.l1_buffer.append(new_observation)
        
        # Update L2 (working) - subsample from L1
        if len(self.l1_buffer) == self.l1_buffer.maxlen:
            l1_summary = torch.stack(list(self.l1_buffer)).mean(dim=0)
            self.l2_buffer.append(l1_summary)
        
        # Update L3 (long-term) - consolidate from L2
        if len(self.l2_buffer) == self.l2_buffer.maxlen:
            l2_summary = torch.stack(list(self.l2_buffer)).mean(dim=0)
            self.l3_memory.store(l2_summary.unsqueeze(0))
    
    def get_context(self, current_observation):
        """현재 관찰을 기반으로 전체 컨텍스트 생성"""
        batch_size = current_observation.shape[0] if current_observation.dim() > 1 else 1
        
        # L1 context (immediate)
        if len(self.l1_buffer) > 0:
            l1_sequence = torch.stack(list(self.l1_buffer)).unsqueeze(0)
            l1_context, _ = self.l1_encoder(l1_sequence)
            l1_context = l1_context[:, -1, :]  # Take last output
        else:
            l1_context = current_observation.unsqueeze(0) if current_observation.dim() == 1 else current_observation
        
        # L2 context (working)
        if len(self.l2_buffer) > 0:
            l2_sequence = torch.stack(list(self.l2_buffer)).unsqueeze(0)
            l2_context, _ = self.l2_encoder(l2_sequence)
            l2_context = l2_context[:, -1, :]
        else:
            l2_context = torch.zeros_like(l1_context)
        
        # L3 context (long-term)
        l3_context, _ = self.l3_memory(current_observation.unsqueeze(0) if current_observation.dim() == 1 else current_observation)
        
        # Attention-weighted fusion
        weights = F.softmax(self.level_attention, dim=0)
        
        # Combine contexts
        combined = torch.cat([
            l1_context * weights[0],
            l2_context * weights[1],
            l3_context * weights[2]
        ], dim=-1)
        
        # Final fusion
        fused_context = self.context_fusion(combined)
        
        return fused_context, {
            'l1': l1_context,
            'l2': l2_context,
            'l3': l3_context,
            'weights': weights
        }
    
    def reset_immediate(self):
        """즉각적인 컨텍스트 리셋 (새 작업 시작)"""
        self.l1_buffer.clear()
    
    def reset_working(self):
        """작업 메모리 리셋 (새 에피소드)"""
        self.l2_buffer.clear()

# 사용 예시
hier_context = HierarchicalContext(feature_dim=768)

# 시뮬레이션
for t in range(100):
    observation = torch.randn(768)
    hier_context.update(observation)
    
    if t % 10 == 0:
        context, levels = hier_context.get_context(observation)
        print(f"Time {t}: Context shape {context.shape}")
        print(f"  Level weights: {levels['weights']}")
```

---

## 🤖 VLA에서의 활용

### 1. Context-Aware VLA Action Generation
```python
class ContextAwareVLA(nn.Module):
    """컨텍스트를 활용한 VLA 행동 생성"""
    def __init__(self, obs_dim=768, action_dim=7):
        super().__init__()
        
        # Memory systems
        self.working_memory = WorkingMemory(memory_size=5, feature_dim=obs_dim)
        self.episodic_memory = EpisodicMemory(episode_length=100, feature_dim=obs_dim)
        self.hierarchical_context = HierarchicalContext(feature_dim=obs_dim)
        
        # Action decoder with context
        self.action_decoder = nn.Sequential(
            nn.Linear(obs_dim * 2, 512),  # current + context
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
    def forward(self, observation, instruction=None):
        # Update hierarchical context
        self.hierarchical_context.update(observation)
        
        # Get context
        context, context_levels = self.hierarchical_context.get_context(observation)
        
        # Retrieve similar past experiences
        similar_episodes = self.episodic_memory.retrieve_similar_episodes(observation, k=3)
        
        # Combine current observation with context
        if instruction is not None:
            # Include language instruction
            combined = torch.cat([observation, instruction, context], dim=-1)
        else:
            combined = torch.cat([observation, context], dim=-1)
        
        # Generate action
        action = self.action_decoder(combined)
        
        # Store in episodic memory
        self.episodic_memory.add_timestep(observation, action)
        
        return action, {
            'context': context,
            'similar_episodes': len(similar_episodes),
            'context_levels': context_levels
        }
```

### 2. Memory-Augmented Decision Making
```python
class MemoryAugmentedPolicy(nn.Module):
    """메모리 증강 정책 네트워크"""
    def __init__(self, state_dim=768, action_dim=7, memory_size=100):
        super().__init__()
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Memory module
        self.memory_module = LongTermMemory(
            memory_bank_size=memory_size, 
            feature_dim=256,
            top_k=5
        )
        
        # Policy network with memory
        self.policy_net = nn.Sequential(
            nn.Linear(256 * 2, 256),  # state + memory
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Value network (for advantage estimation)
        self.value_net = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state, store_memory=True):
        # Encode state
        state_encoded = self.state_encoder(state)
        
        # Retrieve relevant memories
        memory_output, memory_weights = self.memory_module(
            state_encoded, store_new=store_memory
        )
        
        # Combine state and memory
        combined = torch.cat([state_encoded, memory_output], dim=-1)
        
        # Compute action and value
        action = self.policy_net(combined)
        value = self.value_net(combined)
        
        return action, value, memory_weights
    
    def update_memory_with_reward(self, states, rewards):
        """리워드 기반 메모리 중요도 업데이트"""
        # High reward states are more important
        importance_scores = torch.abs(rewards)
        
        for state, importance in zip(states, importance_scores):
            if importance > 0.5:  # Threshold for storage
                state_encoded = self.state_encoder(state)
                self.memory_module.store(state_encoded, importance)
```

---

## 🔬 핵심 개념 정리

### 1. Memory Capacity Analysis
```python
def analyze_memory_capacity(memory_module, test_sequences):
    """메모리 용량 및 성능 분석"""
    results = {
        'capacity': [],
        'retrieval_accuracy': [],
        'forgetting_curve': []
    }
    
    # Test capacity
    for seq_len in [5, 10, 20, 50, 100]:
        sequence = torch.randn(seq_len, 768)
        
        # Store sequence
        for item in sequence:
            memory_module.store(item.unsqueeze(0))
        
        # Test retrieval
        correct_retrievals = 0
        for i, item in enumerate(sequence):
            retrieved, _ = memory_module.retrieve(item.unsqueeze(0))
            similarity = F.cosine_similarity(retrieved, item.unsqueeze(0))
            if similarity > 0.9:
                correct_retrievals += 1
        
        accuracy = correct_retrievals / seq_len
        results['capacity'].append(seq_len)
        results['retrieval_accuracy'].append(accuracy)
    
    return results
```

### 2. Context Window Optimization
```python
def optimize_context_window(task_data, window_sizes=[5, 10, 20, 50]):
    """최적 컨텍스트 윈도우 크기 찾기"""
    best_window = None
    best_performance = -float('inf')
    
    for window_size in window_sizes:
        # Create model with specific window size
        model = WorkingMemory(memory_size=window_size)
        
        # Evaluate on task
        total_reward = 0
        for episode in task_data:
            memory_state = None
            for step in episode:
                output, memory_state, _ = model(step['observation'], memory_state)
                # Compute task-specific reward
                reward = compute_task_reward(output, step['target'])
                total_reward += reward
        
        avg_reward = total_reward / len(task_data)
        
        if avg_reward > best_performance:
            best_performance = avg_reward
            best_window = window_size
    
    return best_window, best_performance
```

---

## 🛠️ 실습 코드

### 완전한 Memory-Enhanced VLA 시스템
```python
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class CompleteMemoryVLA(nn.Module):
    """완전한 메모리 강화 VLA 시스템"""
    def __init__(self, 
                 observation_dim=768,
                 instruction_dim=768,
                 action_dim=7,
                 working_memory_size=7,
                 episodic_buffer_size=100,
                 long_term_bank_size=1000):
        super().__init__()
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(observation_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Instruction encoder
        self.inst_encoder = nn.Sequential(
            nn.Linear(instruction_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Memory components
        self.working_memory = WorkingMemory(
            memory_size=working_memory_size, 
            feature_dim=256
        )
        
        self.episodic_buffer = deque(maxlen=episodic_buffer_size)
        
        self.long_term_memory = LongTermMemory(
            memory_bank_size=long_term_bank_size,
            feature_dim=256,
            top_k=10
        )
        
        # Hierarchical context
        self.context_manager = HierarchicalContext(feature_dim=256)
        
        # Memory-based action generation
        self.action_generator = nn.Sequential(
            nn.Linear(256 * 4, 512),  # obs + inst + working + context
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, action_dim)
        )
        
        # Auxiliary networks
        self.confidence_estimator = nn.Sequential(
            nn.Linear(256 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.success_predictor = nn.Sequential(
            nn.Linear(256 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, observation, instruction, update_memory=True):
        # Encode inputs
        obs_encoded = self.obs_encoder(observation)
        inst_encoded = self.inst_encoder(instruction)
        
        # Get working memory output
        working_output, working_state, _ = self.working_memory(obs_encoded)
        
        # Get hierarchical context
        context, context_info = self.context_manager.get_context(obs_encoded)
        
        # Retrieve from long-term memory
        query = torch.cat([obs_encoded, inst_encoded], dim=-1)
        ltm_output, ltm_weights = self.long_term_memory(query.squeeze(0) if query.dim() > 2 else query)
        
        # Combine all memory sources
        combined_features = torch.cat([
            obs_encoded,
            inst_encoded,
            working_output,
            context
        ], dim=-1)
        
        # Generate action
        action = self.action_generator(combined_features)
        
        # Estimate confidence and success probability
        confidence = self.confidence_estimator(combined_features)
        success_prob = self.success_predictor(combined_features)
        
        # Update memories if specified
        if update_memory:
            # Update episodic buffer
            self.episodic_buffer.append({
                'observation': obs_encoded,
                'instruction': inst_encoded,
                'action': action,
                'timestamp': len(self.episodic_buffer)
            })
            
            # Update context
            self.context_manager.update(obs_encoded.squeeze(0) if obs_encoded.dim() > 1 else obs_encoded)
            
            # Conditionally update long-term memory
            if confidence > 0.8:  # High confidence experiences
                self.long_term_memory.store(obs_encoded)
        
        return {
            'action': action,
            'confidence': confidence,
            'success_probability': success_prob,
            'memory_info': {
                'working_memory': working_state,
                'context_levels': context_info,
                'ltm_retrieval_weights': ltm_weights
            }
        }
    
    def consolidate_memory(self):
        """메모리 통합 및 정리"""
        # Consolidate long-term memory
        self.long_term_memory.consolidate(threshold=0.95)
        
        # Summarize episodic buffer
        if len(self.episodic_buffer) > 50:
            # Extract important episodes
            important_episodes = []
            for episode in self.episodic_buffer:
                if episode.get('reward', 0) > 0.5:
                    important_episodes.append(episode)
            
            # Store in long-term memory
            for episode in important_episodes:
                self.long_term_memory.store(episode['observation'])
            
            # Clear old episodes
            self.episodic_buffer.clear()
            self.episodic_buffer.extend(important_episodes[-20:])  # Keep recent important ones
    
    def reset_episode(self):
        """에피소드 시작 시 리셋"""
        self.context_manager.reset_immediate()
        self.working_memory = WorkingMemory(
            memory_size=self.working_memory.memory_size,
            feature_dim=self.working_memory.feature_dim
        )

# 학습 및 추론 예시
def train_memory_vla():
    model = CompleteMemoryVLA()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Simulated training loop
    for episode in range(100):
        model.reset_episode()
        
        for step in range(50):
            # Simulated inputs
            observation = torch.randn(1, 768)
            instruction = torch.randn(1, 768)
            
            # Forward pass
            output = model(observation, instruction)
            action = output['action']
            confidence = output['confidence']
            
            # Compute loss (example)
            target_action = torch.randn(1, 7)  # Ground truth
            action_loss = nn.MSELoss()(action, target_action)
            
            # Confidence should be high for correct actions
            action_error = (action - target_action).abs().mean()
            confidence_target = torch.exp(-action_error).unsqueeze(0)
            confidence_loss = nn.BCELoss()(confidence, confidence_target)
            
            total_loss = action_loss + 0.1 * confidence_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                print(f"Episode {episode}, Step {step}: Loss {total_loss.item():.4f}, Confidence {confidence.item():.2f}")
        
        # Consolidate memory after episode
        model.consolidate_memory()

# Visualization
def visualize_memory_usage(model, test_sequence):
    """메모리 사용 패턴 시각화"""
    memory_states = []
    context_weights = []
    ltm_weights = []
    
    model.eval()
    with torch.no_grad():
        for t, (obs, inst) in enumerate(test_sequence):
            output = model(obs, inst)
            
            memory_info = output['memory_info']
            memory_states.append(memory_info['working_memory'].cpu().numpy())
            context_weights.append(memory_info['context_levels']['weights'].cpu().numpy())
            ltm_weights.append(memory_info['ltm_retrieval_weights'].cpu().numpy())
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Working memory activation
    axes[0].imshow(np.array(memory_states)[:, 0, :].T, aspect='auto', cmap='hot')
    axes[0].set_title('Working Memory Activation Over Time')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Memory Slot')
    
    # Context level weights
    context_weights = np.array(context_weights)
    axes[1].plot(context_weights[:, 0], label='L1 (Immediate)', alpha=0.7)
    axes[1].plot(context_weights[:, 1], label='L2 (Working)', alpha=0.7)
    axes[1].plot(context_weights[:, 2], label='L3 (Long-term)', alpha=0.7)
    axes[1].set_title('Context Level Attention Weights')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Weight')
    axes[1].legend()
    
    # LTM retrieval patterns
    ltm_weights = np.array(ltm_weights)
    axes[2].imshow(ltm_weights.T, aspect='auto', cmap='viridis')
    axes[2].set_title('Long-term Memory Retrieval Weights')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Memory Bank Index')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create and test model
    model = CompleteMemoryVLA()
    
    # Test single forward pass
    obs = torch.randn(1, 768)
    inst = torch.randn(1, 768)
    output = model(obs, inst)
    
    print("Action shape:", output['action'].shape)
    print("Confidence:", output['confidence'].item())
    print("Success probability:", output['success_probability'].item())
```

---

## 📈 다음 단계

### 1. 고급 메모리 기법
- **Neural Turing Machines**: 완전한 읽기/쓰기 메모리
- **Differentiable Neural Computer**: 구조화된 메모리 접근
- **Transformer-XL**: 긴 컨텍스트 처리

### 2. VLA 특화 개선
- **Skill Memory**: 학습된 스킬 저장 및 재사용
- **Object-Centric Memory**: 객체 중심 메모리 구조
- **Temporal Abstraction**: 다양한 시간 스케일 처리

### 3. 연구 방향
- **Continual Learning**: 지속적 학습과 메모리
- **Meta-Learning**: 빠른 적응을 위한 메모리
- **Neurosymbolic Memory**: 심볼릭 추론과 결합

---

## 💡 핵심 포인트

### ✅ 기억해야 할 것들
1. **메모리 계층**: L1(즉각) → L2(작업) → L3(장기)
2. **Attention 기반 접근**: 관련 메모리 선택적 활용
3. **에피소드 메모리**: 과거 경험에서 학습
4. **컨텍스트 관리**: 상황 인식과 연속성

### ⚠️ 주의사항
1. **메모리 오버헤드**: 저장 공간과 계산 비용
2. **Catastrophic Forgetting**: 새 정보가 기존 정보 덮어쓰기
3. **메모리 품질**: 노이즈나 잘못된 정보 축적

### 🎯 VLA 적용 시
1. **실시간 제약**: 빠른 메모리 접근 필요
2. **안전성**: 중요한 안전 정보 우선 보존
3. **적응성**: 새로운 환경에 빠른 적응

---

**다음 문서**: `10_retrieval_systems.md` - 효율적인 정보 검색 시스템