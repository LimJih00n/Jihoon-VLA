# 🔍 Attention Mechanism

**목표**: Attention의 핵심 개념과 VLA에서의 활용을 코드로 완전 이해

**시간**: 2-3시간

**전제조건**: Neural Networks Basics

---

## 🎯 개발자를 위한 Attention 직관

### Attention = 중요한 부분에 집중하는 메커니즘
```python
# 전통적인 방법: 모든 정보를 똑같이 처리
def traditional_processing(sequence):
    # 모든 요소에 동일한 가중치
    return sum(sequence) / len(sequence)

# Attention 방법: 중요도에 따라 가중 평균
def attention_processing(sequence, query):
    # 1. 각 요소의 중요도 계산
    attention_scores = compute_importance(sequence, query)
    # 2. 중요도로 가중 평균
    weighted_sum = sum(score * value for score, value in zip(attention_scores, sequence))
    return weighted_sum
```

### 핵심 아이디어
```python
attention_concept = {
    "Query (Q)": "지금 찾고자 하는 정보 (질문)",
    "Key (K)": "각 위치의 정보 식별자 (색인)",  
    "Value (V)": "각 위치의 실제 정보 (내용)",
    "과정": "Q와 K를 비교해서 중요도 계산 → V를 중요도로 가중합"
}
```

---

## 🔍 Step-by-Step: Attention 구현

### 1. Scaled Dot-Product Attention (가장 기본)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Attention(Q,K,V) = softmax(QK^T / √d_k)V
    """
    # 1. Q와 K의 내적으로 유사도 계산
    # query: (batch, seq_len, d_model)
    # key: (batch, seq_len, d_model)
    scores = torch.matmul(query, key.transpose(-2, -1))  # (batch, seq_len, seq_len)
    
    # 2. 스케일링 (그라디언트 안정화)
    d_k = query.size(-1)
    scores = scores / math.sqrt(d_k)
    
    # 3. 마스킹 (선택적, 특정 위치 무시)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 4. 소프트맥스로 확률 분포 생성
    attention_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)
    
    # 5. 가중합으로 최종 결과
    output = torch.matmul(attention_weights, value)  # (batch, seq_len, d_model)
    
    return output, attention_weights

# 예시 실행
batch_size, seq_len, d_model = 2, 4, 8

query = torch.randn(batch_size, seq_len, d_model)
key = torch.randn(batch_size, seq_len, d_model)  
value = torch.randn(batch_size, seq_len, d_model)

output, weights = scaled_dot_product_attention(query, key, value)

print(f"입력 크기: Q{query.shape}, K{key.shape}, V{value.shape}")
print(f"출력 크기: {output.shape}")
print(f"Attention 가중치 크기: {weights.shape}")
print(f"가중치 합계 (각 행): {weights.sum(dim=-1)}")  # 모두 1이어야 함
```

### 2. Multi-Head Attention (병렬 처리)
```python
class MultiHeadAttention(nn.Module):
    """
    여러 개의 attention head를 병렬로 실행
    서로 다른 관점에서 정보 수집
    """
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 각 헤드용 선형 변환
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. Q, K, V 생성 및 헤드별 분리
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. 각 헤드에서 어텐션 계산
        attention_output, attention_weights = self.attention(Q, K, V, mask)
        
        # 3. 헤드들을 다시 합치기
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 4. 출력 투영
        output = self.w_o(attention_output)
        
        return output, attention_weights
    
    def attention(self, query, key, value, mask=None):
        """각 헤드에서 스케일드 닷 프로덕트 어텐션"""
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights

# 예시 사용
d_model, n_heads = 256, 8
mha = MultiHeadAttention(d_model, n_heads)

# 입력 데이터
seq_len = 10
x = torch.randn(2, seq_len, d_model)

# Self-attention (Q, K, V가 모두 같은 입력)
output, weights = mha(x, x, x)

print(f"Multi-Head Attention 출력: {output.shape}")
print(f"Attention 가중치: {weights.shape}")  # (batch, n_heads, seq_len, seq_len)
```

### 3. Self-Attention vs Cross-Attention
```python
def demonstrate_attention_types():
    """
    Self-Attention과 Cross-Attention의 차이점 시연
    """
    d_model = 64
    mha = MultiHeadAttention(d_model, 4)
    
    # 문장: "The robot picks up the red cup"
    sentence = torch.randn(1, 6, d_model)  # 6개 단어
    
    # 이미지 특성
    image_features = torch.randn(1, 49, d_model)  # 7x7 이미지 패치
    
    print("=== Self-Attention ===")
    # Self-attention: 문장 내 단어들 간의 관계
    self_attended, self_weights = mha(sentence, sentence, sentence)
    print(f"입력: {sentence.shape} (문장)")
    print(f"출력: {self_attended.shape} (자기 참조 후)")
    print(f"Attention 가중치: {self_weights.shape}")
    
    print("\n=== Cross-Attention ===")  
    # Cross-attention: 문장이 이미지를 참조
    cross_attended, cross_weights = mha(
        query=sentence,      # 문장에서 질문
        key=image_features,  # 이미지에서 검색
        value=image_features # 이미지에서 정보 추출
    )
    print(f"Query: {sentence.shape} (문장)")
    print(f"Key/Value: {image_features.shape} (이미지)")
    print(f"출력: {cross_attended.shape} (이미지 정보가 반영된 문장)")
    print(f"Cross-attention 가중치: {cross_weights.shape}")

demonstrate_attention_types()
```

---

## 🤖 VLA에서의 Attention 활용

### 1. Visual Attention (이미지에서 중요한 부분 찾기)
```python
class VisualAttention(nn.Module):
    """
    로봇이 이미지에서 중요한 영역에 집중
    """
    
    def __init__(self, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 이미지 패치를 처리하는 CNN
        self.patch_encoder = nn.Conv2d(3, feature_dim, kernel_size=16, stride=16)
        
        # Attention mechanism
        self.attention = MultiHeadAttention(feature_dim, 8)
        
        # 공간 위치 인코딩
        self.pos_encoding = nn.Parameter(torch.randn(1, 49, feature_dim))  # 7x7 패치
    
    def forward(self, image, task_query):
        """
        image: (batch, 3, 224, 224)
        task_query: (batch, feature_dim) - 수행할 태스크 정보
        """
        batch_size = image.size(0)
        
        # 1. 이미지를 패치로 분할하고 인코딩
        patches = self.patch_encoder(image)  # (batch, feature_dim, 7, 7)
        patches = patches.flatten(2).transpose(1, 2)  # (batch, 49, feature_dim)
        
        # 2. 위치 인코딩 추가
        patches = patches + self.pos_encoding
        
        # 3. 태스크 기반 어텐션
        # Query: 태스크, Key/Value: 이미지 패치
        task_query = task_query.unsqueeze(1)  # (batch, 1, feature_dim)
        
        attended_features, attention_weights = self.attention(
            query=task_query,
            key=patches,
            value=patches
        )
        
        return attended_features.squeeze(1), attention_weights

# 예시: "빨간 컵을 찾아라" 태스크
visual_attention = VisualAttention()

dummy_image = torch.randn(2, 3, 224, 224)
task_embedding = torch.randn(2, 256)  # "find red cup" 임베딩

attended_visual, attention_map = visual_attention(dummy_image, task_embedding)

print(f"시각적 주의집중 결과: {attended_visual.shape}")
print(f"주의집중 지도: {attention_map.shape}")

# 어텐션 맵 시각화 (개념적)
def visualize_attention_map(attention_weights):
    """
    어텐션 가중치를 7x7 이미지로 시각화
    """
    # attention_weights: (batch, n_heads, 1, 49)
    avg_attention = attention_weights.mean(dim=1).squeeze(1)  # (batch, 49)
    attention_2d = avg_attention.view(-1, 7, 7)  # (batch, 7, 7)
    
    print(f"주의집중 강도 (7x7 그리드):")
    for i in range(attention_2d.size(0)):
        print(f"샘플 {i+1}:")
        print(attention_2d[i].detach().numpy())
        print()

visualize_attention_map(attention_map)
```

### 2. Language Attention (명령어에서 중요한 단어 찾기)
```python
class LanguageAttention(nn.Module):
    """
    자연어 명령에서 중요한 부분 찾기
    """
    
    def __init__(self, vocab_size=10000, embed_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = MultiHeadAttention(embed_dim, 8)
        
        # 위치 인코딩 (Transformer 스타일)
        self.pos_encoding = self.create_positional_encoding(100, embed_dim)
    
    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, tokens, current_state=None):
        """
        tokens: (batch, seq_len) - 토큰화된 명령어
        current_state: (batch, embed_dim) - 현재 로봇 상태 (선택적)
        """
        batch_size, seq_len = tokens.shape
        
        # 1. 토큰 임베딩 + 위치 인코딩
        embedded = self.embedding(tokens)  # (batch, seq_len, embed_dim)
        embedded = embedded + self.pos_encoding[:, :seq_len, :]
        
        if current_state is not None:
            # 2. 현재 상태 기반 어텐션 (어떤 명령어 부분이 현재 상황에 관련있는지)
            state_query = current_state.unsqueeze(1)  # (batch, 1, embed_dim)
            
            attended_lang, attention_weights = self.attention(
                query=state_query,
                key=embedded,
                value=embedded
            )
            
            return attended_lang.squeeze(1), attention_weights
        else:
            # 3. Self-attention (명령어 내 단어들 간의 관계)
            self_attended, attention_weights = self.attention(embedded, embedded, embedded)
            
            # 평균 풀링으로 문장 전체 표현
            sentence_repr = self_attended.mean(dim=1)
            
            return sentence_repr, attention_weights

# 예시 사용
lang_attention = LanguageAttention()

# "Pick up the red cup on the table"
instruction_tokens = torch.tensor([[45, 123, 67, 891, 234, 456, 67, 789]])
robot_state = torch.randn(1, 256)

# 현재 상태 기반 명령어 이해
attended_instruction, lang_attention_weights = lang_attention(
    instruction_tokens, robot_state
)

print(f"주의집중된 명령어: {attended_instruction.shape}")
print(f"언어 어텐션 가중치: {lang_attention_weights.shape}")

# 어떤 단어에 주목했는지 확인
def analyze_language_attention(tokens, attention_weights):
    """
    어떤 단어에 높은 어텐션을 줬는지 분석
    """
    # 단어별 어텐션 점수 (평균)
    word_scores = attention_weights.mean(dim=1).squeeze(1)  # (seq_len,)
    
    # 가상의 단어들
    words = ["pick", "up", "the", "red", "cup", "on", "the", "table"]
    
    print("단어별 어텐션 점수:")
    for word, score in zip(words, word_scores[0]):
        print(f"{word:8}: {score.item():.3f}")

analyze_language_attention(instruction_tokens, lang_attention_weights)
```

### 3. Cross-Modal Attention (Vision ↔ Language)
```python
class CrossModalAttention(nn.Module):
    """
    시각 정보와 언어 정보 간의 상호작용
    """
    
    def __init__(self, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 각 모달리티 처리
        self.vision_proj = nn.Linear(feature_dim, feature_dim)
        self.language_proj = nn.Linear(feature_dim, feature_dim)
        
        # Cross-attention layers
        self.vision_to_lang = MultiHeadAttention(feature_dim, 8)
        self.lang_to_vision = MultiHeadAttention(feature_dim, 8)
        
        # 최종 융합
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, visual_features, language_features):
        """
        visual_features: (batch, num_patches, feature_dim)
        language_features: (batch, seq_len, feature_dim)
        """
        # 프로젝션
        vis_proj = self.vision_proj(visual_features)
        lang_proj = self.language_proj(language_features)
        
        # 1. 언어가 시각 정보 참조 (언어 기반으로 시각 정보 선택)
        lang_attended_vis, _ = self.vision_to_lang(
            query=lang_proj,     # 언어에서 질문
            key=vis_proj,        # 시각에서 검색  
            value=vis_proj       # 시각에서 정보 추출
        )
        
        # 2. 시각이 언어 정보 참조 (시각 기반으로 언어 정보 선택)
        vis_attended_lang, _ = self.lang_to_vision(
            query=vis_proj,      # 시각에서 질문
            key=lang_proj,       # 언어에서 검색
            value=lang_proj      # 언어에서 정보 추출  
        )
        
        # 3. 정보 통합 (평균 풀링 후 융합)
        fused_vis_lang = lang_attended_vis.mean(dim=1)  # (batch, feature_dim)
        fused_lang_vis = vis_attended_lang.mean(dim=1)  # (batch, feature_dim)
        
        # 4. 최종 융합된 표현
        combined = torch.cat([fused_vis_lang, fused_lang_vis], dim=1)
        final_features = self.fusion(combined)
        
        return final_features

# 완전한 VLA 모델 with Attention
class AttentionBasedVLA(nn.Module):
    """
    어텐션을 활용한 완전한 VLA 모델
    """
    
    def __init__(self, feature_dim=256, action_dim=7):
        super().__init__()
        
        # 각 모달리티 인코더
        self.visual_attention = VisualAttention(feature_dim)
        self.language_attention = LanguageAttention(embed_dim=feature_dim)
        
        # 크로스 모달 융합
        self.cross_modal = CrossModalAttention(feature_dim)
        
        # 액션 디코더
        self.action_decoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, images, instructions):
        batch_size = images.size(0)
        
        # 1. 언어 처리 (명령어 이해)
        lang_features, _ = self.language_attention(instructions)
        lang_features = lang_features.unsqueeze(1)  # (batch, 1, feature_dim)
        
        # 2. 시각 처리 (태스크 기반 시각 주의집중)
        vis_features, attention_map = self.visual_attention(images, lang_features.squeeze(1))
        vis_features = vis_features.unsqueeze(1)  # (batch, 1, feature_dim)
        
        # 3. 크로스 모달 융합
        fused_features = self.cross_modal(vis_features, lang_features)
        
        # 4. 액션 예측
        actions = self.action_decoder(fused_features)
        
        return actions, attention_map

# 전체 파이프라인 테스트
attention_vla = AttentionBasedVLA()

dummy_images = torch.randn(2, 3, 224, 224)
dummy_instructions = torch.randint(0, 10000, (2, 8))

predicted_actions, attention_maps = attention_vla(dummy_images, dummy_instructions)

print(f"Attention 기반 VLA 출력:")
print(f"- 예측 액션: {predicted_actions.shape}")
print(f"- 시각 어텐션 맵: {attention_maps.shape}")
print(f"- 액션 값: {predicted_actions}")
```

---

## 🔬 핵심 개념 정리

### 1. Attention의 핵심 아이디어
```python
attention_key_concepts = {
    "선택적_집중": "모든 정보를 동일하게 처리하지 않고 중요한 부분에 집중",
    "동적_가중치": "상황에 따라 어느 부분이 중요한지 동적으로 결정",
    "유연한_정보결합": "서로 다른 소스의 정보를 유연하게 결합",
    "해석가능성": "어느 부분에 주의를 기울였는지 시각화 가능"
}
```

### 2. VLA에서 Attention의 역할
```python
vla_attention_roles = {
    "시각_주의집중": "이미지에서 태스크 관련 영역 찾기",
    "언어_이해": "명령어에서 핵심 키워드 식별",
    "시공간_추론": "여러 시점의 정보를 연결해서 이해",
    "모달리티_융합": "비전과 언어 정보를 효과적으로 결합"
}
```

### 3. Attention vs 기존 방법
```python
attention_vs_traditional = {
    "기존_CNN": {
        "방식": "고정된 필터로 모든 위치 동일 처리",
        "한계": "상황에 따른 적응적 처리 불가"
    },
    
    "기존_RNN": {
        "방식": "순차적으로 정보 처리",
        "한계": "긴 시퀀스에서 정보 손실"
    },
    
    "Attention": {
        "방식": "상황에 맞춰 중요도 동적 계산",
        "장점": "긴 시퀀스, 복잡한 관계 효과적 처리"
    }
}
```

---

## 🛠️ 실습: Attention 시각화

### Attention 가중치 시각화
```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention_weights(attention_weights, words=None, save_path=None):
    """
    Attention 가중치를 히트맵으로 시각화
    """
    # attention_weights: (seq_len, seq_len) 또는 (n_heads, seq_len, seq_len)
    
    if attention_weights.dim() == 3:
        # Multi-head인 경우 평균 계산
        attention_weights = attention_weights.mean(dim=0)
    
    attention_np = attention_weights.detach().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_np, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    
    if words:
        plt.xticks(range(len(words)), words, rotation=45)
        plt.yticks(range(len(words)), words)
        plt.xlabel('Key (참조되는 단어)')
        plt.ylabel('Query (질문하는 단어)')
    
    plt.title('Attention Weights Visualization')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# 예시 데이터로 시각화
def demo_attention_visualization():
    # 간단한 self-attention 예시
    sentence = ["The", "robot", "picks", "up", "red", "cup"]
    seq_len = len(sentence)
    
    # 더미 어텐션 가중치 생성 (실제로는 모델에서 나옴)
    attention_weights = torch.softmax(torch.randn(seq_len, seq_len), dim=-1)
    
    print("문장:", " ".join(sentence))
    print("Attention 가중치 시각화:")
    
    visualize_attention_weights(attention_weights, sentence)

# 실행
demo_attention_visualization()
```

---

## 📈 다음 단계

Attention을 이해했다면:

1. **Transformer Architecture** (`03_transformer_architecture.md`) - Attention의 완전한 활용
2. **Multi-Modal Learning** (`04_multimodal_learning.md`) - Cross-modal attention 심화
3. **Vision Transformers** (`05_vision_encoders.md`) - 이미지에서 Attention 활용

### 추천 실습
```python
recommended_practice = {
    "기초": "간단한 attention 메커니즘 직접 구현",
    "중급": "Multi-head attention으로 간단한 번역 모델",
    "고급": "Vision-Language attention으로 이미지 캡셔닝"
}
```

---

## 💡 핵심 포인트

### 기억해야 할 것
1. **Query, Key, Value**: Attention의 3요소, 각각의 역할 명확히 이해
2. **Softmax의 중요성**: 확률 분포로 만들어서 해석 가능한 가중치 생성
3. **Multi-Head의 장점**: 서로 다른 관점에서 정보 수집
4. **Self vs Cross Attention**: 언제 어떤 것을 사용할지 판단 중요

### VLA 연구에서의 의미
- **핵심 메커니즘**: 현대 VLA 모델의 기본 구성 요소
- **모달리티 결합**: Vision과 Language를 연결하는 핵심 도구
- **해석가능성**: 모델이 왜 그런 결정을 내렸는지 이해 가능
- **성능 향상**: 관련 없는 정보 무시, 중요한 정보에 집중

**다음: Transformer 아키텍처로!** 🚀

---

*Created: 2025-08-24*  
*Time: 2-3 hours*  
*Next: 03_transformer_architecture.md*