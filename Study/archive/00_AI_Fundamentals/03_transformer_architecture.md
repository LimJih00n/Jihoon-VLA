# 🏗️ Transformer Architecture

**목표**: Transformer의 완전한 구조 이해와 VLA에서의 활용

**시간**: 3-4시간

**전제조건**: Neural Networks, Attention Mechanism

---

## 🎯 개발자를 위한 Transformer 직관

### Transformer = Attention 기반 병렬 처리 아키텍처
```python
# 기존 RNN/LSTM: 순차 처리
def rnn_processing(sequence):
    hidden_state = initial_state
    for token in sequence:
        hidden_state = rnn_cell(token, hidden_state)
    return hidden_state

# Transformer: 병렬 처리
def transformer_processing(sequence):
    # 모든 토큰을 동시에 처리
    attended_sequence = multi_head_attention(sequence, sequence, sequence)
    processed_sequence = feed_forward(attended_sequence)
    return processed_sequence
```

### 핵심 혁신
```python
transformer_innovations = {
    "병렬_처리": "모든 위치를 동시에 처리 → 훨씬 빠른 학습",
    "위치_인코딩": "순서 정보를 명시적으로 추가",
    "Residual_Connection": "깊은 네트워크 안정적 학습",
    "Layer_Normalization": "각 층의 출력 정규화"
}
```

---

## 🏗️ Transformer 구조 완전 분해

### 1. 전체 아키텍처 개요
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerBlock(nn.Module):
    """
    Transformer의 기본 블록
    Multi-Head Attention + Feed Forward + Residual Connections
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Feed Forward Network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 1. Multi-Head Self-Attention with Residual Connection
        attended, attention_weights = self.attention(x, x, x, attn_mask=mask)
        x1 = self.norm1(x + self.dropout(attended))
        
        # 2. Feed Forward with Residual Connection  
        fed_forward = self.feed_forward(x1)
        x2 = self.norm2(x1 + fed_forward)
        
        return x2, attention_weights

# 단일 블록 테스트
d_model, n_heads, d_ff = 256, 8, 1024
transformer_block = TransformerBlock(d_model, n_heads, d_ff)

# 입력: (batch, seq_len, d_model)
x = torch.randn(2, 10, d_model)
output, attention = transformer_block(x)

print(f"Transformer Block:")
print(f"입력: {x.shape}")
print(f"출력: {output.shape}")
print(f"Attention: {attention.shape}")
```

### 2. 위치 인코딩 (Positional Encoding)
```python
class PositionalEncoding(nn.Module):
    """
    위치 정보를 sine/cosine 함수로 인코딩
    Transformer는 순서를 모르므로 위치 정보 필요
    """
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # 위치 인코딩 테이블 생성
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # div_term: 1/10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # 짝수 인덱스: sin, 홀수 인덱스: cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # 학습되지 않는 파라미터로 등록
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0), :]
        return x

# 위치 인코딩 시각화
def visualize_positional_encoding():
    pe = PositionalEncoding(256, 100)
    
    # 처음 100개 위치의 인코딩
    dummy_input = torch.zeros(100, 1, 256)
    encoded = pe(dummy_input)
    
    # 첫 번째 배치의 위치 인코딩 시각화
    pos_encoding = encoded[:, 0, :].detach().numpy()
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.imshow(pos_encoding.T, cmap='RdYlBu', aspect='auto')
    plt.colorbar()
    plt.xlabel('Position')
    plt.ylabel('Encoding Dimension')
    plt.title('Positional Encoding Visualization')
    plt.show()

# visualize_positional_encoding()  # 실행하면 그래프 출력
```

### 3. 완전한 Transformer 인코더
```python
class TransformerEncoder(nn.Module):
    """
    여러 Transformer 블록을 쌓은 완전한 인코더
    """
    
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len=5000, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 토큰 임베딩
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 위치 인코딩
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # 여러 Transformer 블록
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        # 1. 토큰 임베딩 + 스케일링
        embedded = self.embedding(src) * math.sqrt(self.d_model)  # (batch, seq, d_model)
        
        # 2. 위치 인코딩 추가
        # PyTorch MultiheadAttention expects (seq, batch, d_model)
        embedded = embedded.transpose(0, 1)  # (seq, batch, d_model)
        embedded = self.positional_encoding(embedded)
        embedded = embedded.transpose(0, 1)  # (batch, seq, d_model)
        embedded = self.dropout(embedded)
        
        # 3. Transformer 블록들 통과
        x = embedded
        attention_weights = []
        
        for transformer_block in self.transformer_blocks:
            x, attention = transformer_block(x, src_mask)
            attention_weights.append(attention)
        
        return x, attention_weights

# 완전한 인코더 테스트
encoder = TransformerEncoder(
    vocab_size=10000,
    d_model=256,
    n_heads=8,
    d_ff=1024,
    n_layers=6
)

# 입력 토큰 시퀀스
input_tokens = torch.randint(0, 10000, (2, 12))  # batch_size=2, seq_len=12
encoded_output, all_attention = encoder(input_tokens)

print(f"Transformer Encoder:")
print(f"입력 토큰: {input_tokens.shape}")
print(f"인코딩된 출력: {encoded_output.shape}")
print(f"레이어별 Attention: {len(all_attention)} layers")
```

### 4. 마스킹 (Masking) 메커니즘
```python
def create_padding_mask(seq, pad_token=0):
    """
    패딩 토큰을 무시하는 마스크 생성
    """
    # seq: (batch, seq_len)
    return seq != pad_token

def create_look_ahead_mask(seq_len):
    """
    미래 토큰을 보지 못하게 하는 마스크 (디코더용)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0  # True는 보여줄 부분, False는 마스크할 부분

def demonstrate_masking():
    """
    마스킹의 효과 시연
    """
    seq_len = 5
    
    print("=== Look-Ahead Mask ===")
    look_ahead_mask = create_look_ahead_mask(seq_len)
    print("현재 위치에서 볼 수 있는 토큰 (True), 마스크된 토큰 (False):")
    print(look_ahead_mask.int())
    
    print("\n=== Padding Mask ===")
    # 예시: [1, 25, 463, 78, 0, 0] (0은 패딩)
    sequence = torch.tensor([[1, 25, 463, 78, 0, 0],
                           [15, 234, 0, 0, 0, 0]])
    padding_mask = create_padding_mask(sequence)
    print("실제 토큰 (True), 패딩 (False):")
    print(padding_mask)

demonstrate_masking()
```

---

## 🤖 VLA를 위한 특별한 Transformer 변형

### 1. Vision Transformer (ViT) for VLA
```python
class VisionTransformer(nn.Module):
    """
    이미지를 패치로 나누고 Transformer로 처리
    VLA에서 시각 정보 처리용
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 d_model=768, n_heads=12, n_layers=12):
        super().__init__()
        
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 이미지 패치를 벡터로 변환
        self.patch_embedding = nn.Conv2d(
            in_channels, d_model, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # 클래스 토큰 (전체 이미지 정보)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 위치 임베딩 (패치 위치 정보)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patches + 1, d_model)
        )
        
        # Transformer 블록들
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model*4)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. 이미지를 패치로 분할 및 임베딩
        # x: (batch, 3, 224, 224) -> patches: (batch, d_model, 14, 14)
        patches = self.patch_embedding(x)  
        patches = patches.flatten(2).transpose(1, 2)  # (batch, n_patches, d_model)
        
        # 2. 클래스 토큰 추가
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)  # (batch, n_patches+1, d_model)
        
        # 3. 위치 임베딩 추가
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # 4. Transformer 블록들 통과
        for transformer_block in self.transformer_blocks:
            x, _ = transformer_block(x)
        
        # 5. 클래스 토큰 반환 (전체 이미지 표현)
        return x[:, 0]  # (batch, d_model)

# Vision Transformer 테스트
vit = VisionTransformer(img_size=224, patch_size=16, d_model=768)
dummy_image = torch.randn(2, 3, 224, 224)
image_features = vit(dummy_image)

print(f"Vision Transformer:")
print(f"이미지 입력: {dummy_image.shape}")
print(f"이미지 특성: {image_features.shape}")
```

### 2. Cross-Modal Transformer for VLA
```python
class CrossModalTransformer(nn.Module):
    """
    비전과 언어를 결합하는 Cross-Modal Transformer
    VLA의 핵심 구성 요소
    """
    
    def __init__(self, vision_dim=768, language_dim=512, d_model=512, 
                 n_heads=8, n_layers=6):
        super().__init__()
        
        # 각 모달리티를 공통 차원으로 투영
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.language_proj = nn.Linear(language_dim, d_model)
        
        # 모달리티 구분을 위한 타입 임베딩
        self.modal_type_embedding = nn.Embedding(2, d_model)  # 0: vision, 1: language
        
        # Cross-Modal Transformer 블록들
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model*4)
            for _ in range(n_layers)
        ])
        
        # 액션 예측 헤드
        self.action_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 7)  # 7-DOF 로봇 액션
        )
    
    def forward(self, vision_features, language_features):
        """
        vision_features: (batch, n_patches, vision_dim)
        language_features: (batch, seq_len, language_dim)
        """
        batch_size = vision_features.size(0)
        
        # 1. 각 모달리티를 공통 차원으로 투영
        vis_projected = self.vision_proj(vision_features)  # (batch, n_patches, d_model)
        lang_projected = self.language_proj(language_features)  # (batch, seq_len, d_model)
        
        # 2. 모달리티 타입 임베딩 추가
        n_patches = vis_projected.size(1)
        seq_len = lang_projected.size(1)
        
        vis_type = torch.zeros(batch_size, n_patches, dtype=torch.long, device=vis_projected.device)
        lang_type = torch.ones(batch_size, seq_len, dtype=torch.long, device=lang_projected.device)
        
        vis_projected += self.modal_type_embedding(vis_type)
        lang_projected += self.modal_type_embedding(lang_type)
        
        # 3. 비전과 언어 토큰 결합
        combined_features = torch.cat([vis_projected, lang_projected], dim=1)
        # (batch, n_patches + seq_len, d_model)
        
        # 4. Cross-Modal Transformer 처리
        for transformer_block in self.transformer_blocks:
            combined_features, _ = transformer_block(combined_features)
        
        # 5. Global average pooling 후 액션 예측
        global_features = combined_features.mean(dim=1)  # (batch, d_model)
        actions = self.action_head(global_features)
        
        return actions

# Cross-Modal Transformer 테스트
cross_modal = CrossModalTransformer()

# 더미 입력
vision_feat = torch.randn(2, 196, 768)  # 14x14 = 196 패치
language_feat = torch.randn(2, 8, 512)   # 8개 단어

predicted_actions = cross_modal(vision_feat, language_feat)

print(f"Cross-Modal Transformer:")
print(f"비전 입력: {vision_feat.shape}")  
print(f"언어 입력: {language_feat.shape}")
print(f"예측 액션: {predicted_actions.shape}")
```

### 3. 완전한 VLA Transformer 모델
```python
class VLATransformer(nn.Module):
    """
    완전한 Vision-Language-Action Transformer
    """
    
    def __init__(self, vocab_size=10000, img_size=224, patch_size=16,
                 d_model=512, n_heads=8, n_layers=6, action_dim=7):
        super().__init__()
        
        # Vision Transformer (이미지 인코더)
        self.vision_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers//2  # 절반은 비전, 절반은 크로스모달
        )
        
        # Language Transformer (언어 인코더)
        self.language_encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_model*4,
            n_layers=n_layers//2
        )
        
        # Cross-Modal Fusion
        self.cross_modal_fusion = CrossModalTransformer(
            vision_dim=d_model,
            language_dim=d_model,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers//2
        )
    
    def forward(self, images, instruction_tokens):
        """
        완전한 VLA forward pass
        """
        # 1. 비전 인코딩
        vision_cls = self.vision_encoder(images)  # (batch, d_model)
        # 패치별 특성도 필요하면 전체 시퀀스 사용
        
        # 2. 언어 인코딩  
        language_encoded, _ = self.language_encoder(instruction_tokens)
        # (batch, seq_len, d_model)
        
        # 3. 크로스 모달 융합을 위해 차원 맞춤
        vision_expanded = vision_cls.unsqueeze(1)  # (batch, 1, d_model)
        
        # 4. 최종 액션 예측
        actions = self.cross_modal_fusion(vision_expanded, language_encoded)
        
        return actions

# 전체 VLA Transformer 테스트
vla_transformer = VLATransformer()

dummy_images = torch.randn(2, 3, 224, 224)
dummy_instructions = torch.randint(0, 10000, (2, 8))

final_actions = vla_transformer(dummy_images, dummy_instructions)

print(f"완전한 VLA Transformer:")
print(f"이미지: {dummy_images.shape}")
print(f"명령어: {dummy_instructions.shape}")  
print(f"최종 액션: {final_actions.shape}")
print(f"액션 값: {final_actions}")
```

---

## 🔬 Transformer vs 다른 아키텍처 비교

### 1. RNN/LSTM vs Transformer
```python
comparison_rnn_transformer = {
    "처리_방식": {
        "RNN/LSTM": "순차적 처리 (t=1 → t=2 → t=3)",
        "Transformer": "병렬 처리 (모든 t 동시에)"
    },
    
    "장거리_의존성": {
        "RNN/LSTM": "정보가 점차 희석됨 (Vanishing gradient)",
        "Transformer": "직접 연결 (Attention으로 모든 위치 접근)"
    },
    
    "학습_속도": {
        "RNN/LSTM": "느림 (순차 처리)",
        "Transformer": "빠름 (병렬 처리 가능)"
    },
    
    "메모리": {
        "RNN/LSTM": "적음",
        "Transformer": "많음 (모든 위치 간 attention 계산)"
    }
}
```

### 2. CNN vs Vision Transformer
```python
comparison_cnn_vit = {
    "귀납적_편향": {
        "CNN": "국소성 (locality), 이동 불변성 (translation invariance)",
        "ViT": "없음 (데이터로부터 모든 것을 학습)"
    },
    
    "수용_영역": {
        "CNN": "점진적 확장 (작은 필터 → 큰 수용 영역)",
        "ViT": "글로벌 (첫 번째 층부터 전체 이미지 접근)"
    },
    
    "데이터_요구량": {
        "CNN": "적음 (귀납적 편향 덕분)",
        "ViT": "많음 (편향 없이 모든 것을 학습해야 함)"
    }
}
```

---

## 🛠️ 실습: Transformer 훈련 및 분석

### 1. 간단한 시퀀스-투-시퀀스 작업
```python
def create_copy_task_data(batch_size=32, seq_len=10, vocab_size=100):
    """
    간단한 복사 작업: 입력 시퀀스를 그대로 출력
    Transformer가 잘 작동하는지 확인용
    """
    # 랜덤 시퀀스 생성
    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    tgt = src.clone()  # 복사 작업이므로 타겟은 소스와 동일
    
    return src, tgt

def train_transformer_copy_task():
    """
    Transformer로 복사 작업 학습
    """
    # 모델 초기화
    model = TransformerEncoder(
        vocab_size=100,
        d_model=128,
        n_heads=4,
        d_ff=256,
        n_layers=2
    )
    
    # 출력 헤드 추가 (vocab_size 차원으로 예측)
    output_head = nn.Linear(128, 100)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(output_head.parameters()), 
        lr=0.001
    )
    criterion = nn.CrossEntropyLoss()
    
    print("복사 작업 학습 시작...")
    
    for epoch in range(100):
        # 데이터 생성
        src, tgt = create_copy_task_data(batch_size=32, seq_len=8)
        
        # Forward pass
        encoded, _ = model(src)  # (batch, seq_len, d_model)
        logits = output_head(encoded)  # (batch, seq_len, vocab_size)
        
        # 손실 계산
        loss = criterion(
            logits.view(-1, 100),  # (batch*seq_len, vocab_size)
            tgt.view(-1)           # (batch*seq_len,)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # 테스트
    with torch.no_grad():
        test_src, test_tgt = create_copy_task_data(batch_size=1, seq_len=5)
        encoded, _ = model(test_src)
        logits = output_head(encoded)
        predictions = logits.argmax(dim=-1)
        
        print(f"\n테스트 결과:")
        print(f"입력:    {test_src[0].tolist()}")
        print(f"정답:    {test_tgt[0].tolist()}")
        print(f"예측:    {predictions[0].tolist()}")
        print(f"정확도:  {(predictions[0] == test_tgt[0]).float().mean().item():.2f}")

# 실행
train_transformer_copy_task()
```

### 2. Attention 패턴 분석
```python
def analyze_attention_patterns(model, input_sequence):
    """
    Transformer의 attention 패턴 분석
    """
    model.eval()
    with torch.no_grad():
        _, attention_weights = model(input_sequence)
    
    # 각 레이어의 attention 분석
    for layer_idx, attention in enumerate(attention_weights):
        print(f"\n=== Layer {layer_idx + 1} ===")
        
        # attention: (batch, n_heads, seq_len, seq_len)
        batch_idx = 0  # 첫 번째 샘플만 분석
        
        for head_idx in range(attention.size(1)):
            head_attention = attention[batch_idx, head_idx]  # (seq_len, seq_len)
            
            # 각 위치가 어디에 가장 많이 주의를 기울이는지
            max_attention_pos = head_attention.argmax(dim=-1)
            
            print(f"Head {head_idx}: 각 위치의 최대 attention 대상")
            for pos, target in enumerate(max_attention_pos):
                print(f"  Position {pos} → Position {target.item()}")

# 분석 실행 예시
def demo_attention_analysis():
    model = TransformerEncoder(vocab_size=100, d_model=64, n_heads=2, d_ff=128, n_layers=2)
    test_input = torch.randint(1, 100, (1, 6))  # 배치 크기 1, 시퀀스 길이 6
    
    analyze_attention_patterns(model, test_input)

demo_attention_analysis()
```

---

## 📈 VLA에서 Transformer 활용 전략

### 1. 계층적 처리 (Hierarchical Processing)
```python
vla_transformer_strategy = {
    "Low_Level": {
        "역할": "즉각적인 센서-액션 매핑",
        "아키텍처": "작은 Transformer (2-4 layers)",
        "처리": "현재 관찰 → 즉각적 액션"
    },
    
    "Mid_Level": {
        "역할": "태스크 수준 추론",
        "아키텍처": "중간 Transformer (6-8 layers)",  
        "처리": "명령어 + 관찰 → 서브 골"
    },
    
    "High_Level": {
        "역할": "장기 계획 및 추론",
        "아키텍처": "큰 Transformer (12+ layers)",
        "처리": "복잡한 명령어 → 전체 계획"
    }
}
```

### 2. 효율성 최적화
```python
efficiency_optimizations = {
    "Attention_Sparsity": {
        "문제": "O(n²) 복잡도로 긴 시퀀스에서 느림",
        "해결": "Sparse attention, Local attention",
        "VLA_적용": "최근 관찰에만 집중"
    },
    
    "Model_Distillation": {
        "문제": "큰 모델은 실시간 로봇 제어에 부적합",
        "해결": "큰 모델 → 작은 모델 지식 증류",
        "VLA_적용": "GPT-4 teacher → 로봇용 student"
    },
    
    "Dynamic_Inference": {
        "문제": "모든 상황에 동일한 계산량 낭비",
        "해결": "상황에 따른 동적 깊이 조절",
        "VLA_적용": "긴급상황 → 빠른 추론, 복잡한 계획 → 깊은 추론"
    }
}
```

---

## 💡 핵심 포인트

### 기억해야 할 것
1. **병렬 처리의 힘**: RNN 대비 훨씬 빠른 학습과 추론
2. **Attention의 활용**: 입력의 모든 부분에 직접 접근 가능
3. **Residual Connection**: 깊은 네트워크를 안정적으로 학습
4. **위치 인코딩**: 순서 정보를 명시적으로 제공

### VLA 연구에서의 의미
- **기본 아키텍처**: 현대 VLA 모델의 표준 구조
- **모달리티 통합**: Cross-attention으로 비전과 언어 효과적 결합
- **확장성**: 더 많은 데이터와 계산으로 성능 향상 가능
- **전이 학습**: 사전 학습된 Transformer를 VLA에 활용

### 다음 단계
1. **Multi-Modal Learning** (`04_multimodal_learning.md`) - Transformer 기반 멀티모달 시스템
2. **Vision Encoders** (`05_vision_encoders.md`) - ViT 심화 학습  
3. **Language Models** (`06_language_models.md`) - GPT 스타일 Transformer

**다음: Multi-Modal Learning으로!** 🚀

---

*Created: 2025-08-24*  
*Time: 3-4 hours*  
*Next: 04_multimodal_learning.md*