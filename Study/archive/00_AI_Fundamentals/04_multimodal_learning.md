# 🎯 Multi-Modal Learning: Vision과 Language의 융합

**목표**: Vision과 Language를 결합하는 멀티모달 학습의 핵심 원리 이해 및 VLA 적용  
**시간**: 3-4시간  
**전제조건**: 01_neural_networks_basics.md, 02_attention_mechanism.md, 03_transformer_architecture.md  

---

## 🎯 개발자를 위한 직관적 이해

### 멀티모달 학습이란?
```python
# 단일 모달 vs 멀티모달
single_modal = {
    "vision_only": "이미지만 보고 분류",
    "text_only": "텍스트만 읽고 이해"
}

multi_modal = {
    "vision_language": "이미지를 보고 + 텍스트 설명을 이해",
    "example": "빨간 공을 집어라" → 시각(빨간 공 찾기) + 언어(명령 이해)
}
```

### 왜 중요한가?
- **로봇 제어**: 자연어 명령 + 시각적 환경 이해
- **풍부한 표현**: 각 모달의 한계를 상호 보완
- **Zero-shot 일반화**: 보지 못한 조합도 이해 가능

---

## 🏗️ 기본 구조 및 구현

### 1. Early Fusion vs Late Fusion
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyFusion(nn.Module):
    """입력 단계에서 모달리티 결합"""
    def __init__(self, vision_dim=768, text_dim=768, hidden_dim=512):
        super().__init__()
        # 먼저 결합 후 처리
        self.fusion = nn.Linear(vision_dim + text_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256)
        )
    
    def forward(self, vision_features, text_features):
        # 초기 단계에서 concatenate
        combined = torch.cat([vision_features, text_features], dim=-1)
        fused = self.fusion(combined)
        output = self.mlp(fused)
        return output

class LateFusion(nn.Module):
    """각 모달리티 독립 처리 후 결합"""
    def __init__(self, vision_dim=768, text_dim=768, hidden_dim=512):
        super().__init__()
        # 각각 독립적으로 처리
        self.vision_encoder = nn.Linear(vision_dim, hidden_dim)
        self.text_encoder = nn.Linear(text_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256)
        )
    
    def forward(self, vision_features, text_features):
        # 독립적 인코딩
        vision_encoded = self.vision_encoder(vision_features)
        text_encoded = self.text_encoder(text_features)
        # 나중에 결합
        combined = torch.cat([vision_encoded, text_encoded], dim=-1)
        output = self.fusion(combined)
        return output
```

### 2. Cross-Modal Attention
```python
class CrossModalAttention(nn.Module):
    """Vision과 Language 간 상호 attention"""
    def __init__(self, d_model=768, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Vision이 Language를 참조
        self.vision_to_text = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # Language가 Vision을 참조
        self.text_to_vision = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # Layer Norm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, vision_features, text_features):
        # vision_features: [batch, n_patches, d_model]
        # text_features: [batch, seq_len, d_model]
        
        # Vision이 Text를 참조하여 업데이트
        vision_attended, _ = self.vision_to_text(
            query=vision_features,
            key=text_features,
            value=text_features
        )
        vision_out = self.ln1(vision_features + vision_attended)
        
        # Text가 Vision을 참조하여 업데이트
        text_attended, _ = self.text_to_vision(
            query=text_features,
            key=vision_features,
            value=vision_features
        )
        text_out = self.ln2(text_features + text_attended)
        
        return vision_out, text_out

# 사용 예시
cross_attn = CrossModalAttention()
batch_size = 2
n_patches = 196  # 14x14 patches
seq_len = 20
d_model = 768

vision_input = torch.randn(batch_size, n_patches, d_model)
text_input = torch.randn(batch_size, seq_len, d_model)

vision_out, text_out = cross_attn(vision_input, text_input)
print(f"Vision output: {vision_out.shape}")  # [2, 196, 768]
print(f"Text output: {text_out.shape}")      # [2, 20, 768]
```

### 3. Contrastive Learning (CLIP 스타일)
```python
class ContrastiveMultiModal(nn.Module):
    """CLIP 스타일의 contrastive learning"""
    def __init__(self, vision_encoder, text_encoder, embed_dim=512, temperature=0.07):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        # Projection heads
        self.vision_proj = nn.Linear(vision_encoder.output_dim, embed_dim)
        self.text_proj = nn.Linear(text_encoder.output_dim, embed_dim)
        
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        
    def forward(self, images, texts):
        # 각 모달리티 인코딩
        vision_features = self.vision_encoder(images)
        text_features = self.text_encoder(texts)
        
        # 공통 embedding space로 projection
        vision_embeds = self.vision_proj(vision_features)
        text_embeds = self.text_proj(text_features)
        
        # L2 정규화
        vision_embeds = F.normalize(vision_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        # Cosine similarity 계산
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * vision_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text
    
    def compute_loss(self, logits_per_image, logits_per_text):
        """InfoNCE loss 계산"""
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size).to(logits_per_image.device)
        
        # Cross entropy loss
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        
        return (loss_i2t + loss_t2i) / 2
```

### 4. Multi-Modal Transformer
```python
class MultiModalTransformer(nn.Module):
    """Vision과 Language를 함께 처리하는 Transformer"""
    def __init__(self, d_model=768, n_heads=12, n_layers=6):
        super().__init__()
        self.d_model = d_model
        
        # Modal-specific embeddings
        self.vision_embed = nn.Linear(768, d_model)
        self.text_embed = nn.Embedding(50000, d_model)  # vocab size 50000
        
        # Positional embeddings
        self.vision_pos_embed = nn.Parameter(torch.randn(1, 196, d_model))
        self.text_pos_embed = nn.Parameter(torch.randn(1, 512, d_model))
        
        # Modal type embeddings
        self.vision_type_embed = nn.Parameter(torch.randn(1, 1, d_model))
        self.text_type_embed = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        
        self.ln_final = nn.LayerNorm(d_model)
        
    def forward(self, vision_features, text_ids, text_mask=None):
        batch_size = vision_features.shape[0]
        
        # Vision embedding
        vision_embeds = self.vision_embed(vision_features)
        vision_embeds = vision_embeds + self.vision_pos_embed[:, :vision_embeds.shape[1]]
        vision_embeds = vision_embeds + self.vision_type_embed
        
        # Text embedding
        text_embeds = self.text_embed(text_ids)
        text_embeds = text_embeds + self.text_pos_embed[:, :text_embeds.shape[1]]
        text_embeds = text_embeds + self.text_type_embed
        
        # Concatenate modalities
        combined = torch.cat([vision_embeds, text_embeds], dim=1)
        
        # Create attention mask
        if text_mask is not None:
            vision_mask = torch.ones(batch_size, vision_embeds.shape[1]).to(text_mask.device)
            combined_mask = torch.cat([vision_mask, text_mask], dim=1)
        else:
            combined_mask = None
        
        # Process through transformer
        for layer in self.layers:
            combined = layer(combined, mask=combined_mask)
        
        output = self.ln_final(combined)
        
        # Split back into modalities if needed
        vision_out = output[:, :vision_embeds.shape[1]]
        text_out = output[:, vision_embeds.shape[1]:]
        
        return output, vision_out, text_out

class TransformerBlock(nn.Module):
    """Single transformer block"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.ln1(x + attn_out)
        
        # MLP
        mlp_out = self.mlp(x)
        x = self.ln2(x + mlp_out)
        
        return x
```

---

## 🤖 VLA에서의 활용

### 1. Vision-Language-Action 결합
```python
class VLAMultiModal(nn.Module):
    """VLA를 위한 멀티모달 처리"""
    def __init__(self, vision_dim=768, language_dim=768, action_dim=7):
        super().__init__()
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(d_model=768)
        
        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
    def forward(self, vision_obs, language_instruction):
        # Cross-modal interaction
        vision_attended, language_attended = self.cross_attention(
            vision_obs, language_instruction
        )
        
        # Pool features
        vision_pooled = vision_attended.mean(dim=1)  # [batch, d_model]
        language_pooled = language_attended.mean(dim=1)  # [batch, d_model]
        
        # Concatenate and decode to actions
        combined = torch.cat([vision_pooled, language_pooled], dim=-1)
        actions = self.action_decoder(combined)
        
        return actions

# 사용 예시
vla_model = VLAMultiModal()

# Robot observation (from camera)
vision_obs = torch.randn(1, 196, 768)  # [batch, patches, dim]

# Language instruction 
instruction = torch.randn(1, 20, 768)  # [batch, seq_len, dim]

# Predict robot actions
actions = vla_model(vision_obs, instruction)
print(f"Predicted actions: {actions.shape}")  # [1, 7] (x,y,z,rx,ry,rz,gripper)
```

### 2. Context-Aware Multimodal Fusion
```python
class ContextAwareMultiModal(nn.Module):
    """Context를 고려한 멀티모달 융합"""
    def __init__(self, d_model=768, context_size=10):
        super().__init__()
        self.context_size = context_size
        
        # Context memory
        self.context_memory = []
        
        # Attention mechanisms
        self.self_attention = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.cross_attention = CrossModalAttention(d_model)
        
        # Context encoder
        self.context_encoder = nn.LSTM(d_model * 2, d_model, batch_first=True)
        
    def forward(self, vision, language, update_context=True):
        # Current frame processing
        vision_current, language_current = self.cross_attention(vision, language)
        
        # Combine current features
        current_features = torch.cat([
            vision_current.mean(dim=1),
            language_current.mean(dim=1)
        ], dim=-1)
        
        # Process with context
        if len(self.context_memory) > 0:
            context_tensor = torch.stack(self.context_memory, dim=1)
            # Add current to context
            full_context = torch.cat([context_tensor, current_features.unsqueeze(1)], dim=1)
            
            # Process through LSTM
            context_encoded, _ = self.context_encoder(full_context)
            output = context_encoded[:, -1]  # Take last output
        else:
            output, _ = self.context_encoder(current_features.unsqueeze(1))
            output = output.squeeze(1)
        
        # Update context memory
        if update_context:
            self.context_memory.append(current_features.detach())
            if len(self.context_memory) > self.context_size:
                self.context_memory.pop(0)
        
        return output
```

### 3. Hierarchical Multimodal Processing
```python
class HierarchicalMultiModal(nn.Module):
    """계층적 멀티모달 처리 (π₀ 스타일)"""
    def __init__(self):
        super().__init__()
        
        # Low-level: Direct sensor fusion
        self.low_level_fusion = EarlyFusion(vision_dim=768, text_dim=768)
        
        # Mid-level: Cross-modal reasoning
        self.mid_level = CrossModalAttention(d_model=512)
        
        # High-level: Abstract planning
        self.high_level = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True),
            num_layers=3
        )
        
        # Final action decoder
        self.action_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )
        
    def forward(self, vision, language):
        # Low-level processing
        low_features = self.low_level_fusion(vision.mean(dim=1), language.mean(dim=1))
        
        # Mid-level processing
        vision_mid = F.interpolate(vision.transpose(1, 2), size=512).transpose(1, 2)
        language_mid = F.interpolate(language.transpose(1, 2), size=512).transpose(1, 2)
        vision_refined, language_refined = self.mid_level(vision_mid, language_mid)
        
        # Combine low and mid features
        combined = low_features.unsqueeze(1)  # [batch, 1, 256]
        
        # High-level reasoning
        high_features = self.high_level(combined)
        
        # Generate actions
        actions = self.action_head(high_features.squeeze(1))
        
        return actions
```

---

## 🔬 핵심 개념 정리

### 1. Alignment Quality Metrics
```python
def compute_alignment_metrics(vision_embeds, text_embeds):
    """멀티모달 정렬 품질 측정"""
    # Normalize embeddings
    vision_norm = F.normalize(vision_embeds, dim=-1)
    text_norm = F.normalize(text_embeds, dim=-1)
    
    # Compute similarity matrix
    similarity = torch.matmul(vision_norm, text_norm.t())
    
    # Metrics
    metrics = {
        # Mean similarity of matched pairs
        "matched_similarity": torch.diagonal(similarity).mean().item(),
        
        # Mean similarity of all pairs
        "mean_similarity": similarity.mean().item(),
        
        # Retrieval accuracy (top-1)
        "retrieval_acc": (similarity.argmax(dim=1) == torch.arange(len(similarity)).to(similarity.device)).float().mean().item()
    }
    
    return metrics
```

### 2. Modality Gap
```python
def measure_modality_gap(vision_embeds, text_embeds):
    """모달리티 간 거리 측정"""
    # Compute centroids
    vision_center = vision_embeds.mean(dim=0)
    text_center = text_embeds.mean(dim=0)
    
    # Modality gap
    gap = torch.norm(vision_center - text_center).item()
    
    # Within-modality variance
    vision_var = torch.var(vision_embeds, dim=0).mean().item()
    text_var = torch.var(text_embeds, dim=0).mean().item()
    
    return {
        "modality_gap": gap,
        "vision_variance": vision_var,
        "text_variance": text_var,
        "gap_to_variance_ratio": gap / (vision_var + text_var)
    }
```

---

## 🛠️ 실습 코드

### 완전한 멀티모달 VLA 예시
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

class SimpleMultiModalVLA(nn.Module):
    """간단한 멀티모달 VLA 구현"""
    def __init__(self, 
                 vision_backbone="resnet50",
                 language_model="bert-base",
                 action_dim=7):
        super().__init__()
        
        # Vision encoder (pretrained ResNet)
        import torchvision.models as models
        resnet = models.resnet50(pretrained=True)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.vision_proj = nn.Linear(2048, 768)
        
        # Language encoder (simple embedding)
        self.language_encoder = nn.Embedding(10000, 768)
        self.language_lstm = nn.LSTM(768, 768, batch_first=True)
        
        # Cross-modal fusion
        self.cross_attention = CrossModalAttention(d_model=768, n_heads=8)
        
        # Action prediction
        self.action_mlp = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, action_dim)
        )
        
    def forward(self, image, instruction_ids):
        # Vision encoding
        with torch.no_grad():  # Freeze vision backbone
            vision_features = self.vision_encoder(image)
        vision_features = vision_features.squeeze(-1).squeeze(-1)
        vision_features = self.vision_proj(vision_features)
        vision_features = vision_features.unsqueeze(1)  # [batch, 1, 768]
        
        # Language encoding
        language_embeds = self.language_encoder(instruction_ids)
        language_features, _ = self.language_lstm(language_embeds)
        
        # Cross-modal attention
        vision_attended, language_attended = self.cross_attention(
            vision_features, language_features
        )
        
        # Pool and concatenate
        vision_pooled = vision_attended.squeeze(1)
        language_pooled = language_attended.mean(dim=1)
        combined = torch.cat([vision_pooled, language_pooled], dim=-1)
        
        # Predict actions
        actions = self.action_mlp(combined)
        
        return actions

# 학습 루프 예시
def train_multimodal_vla(model, dataloader, epochs=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            images = batch['image']
            instructions = batch['instruction']
            target_actions = batch['action']
            
            # Forward pass
            predicted_actions = model(images, instructions)
            loss = criterion(predicted_actions, target_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# 추론 예시
def inference_example():
    model = SimpleMultiModalVLA()
    model.eval()
    
    # 가상의 입력
    image = torch.randn(1, 3, 224, 224)
    instruction = torch.tensor([[1, 5, 10, 3, 0, 0, 0]])  # "pick red cube"
    
    with torch.no_grad():
        actions = model(image, instruction)
    
    print(f"Predicted robot actions: {actions}")
    print(f"  Position (x,y,z): {actions[0, :3]}")
    print(f"  Rotation (rx,ry,rz): {actions[0, 3:6]}")
    print(f"  Gripper: {actions[0, 6]}")

# 실행
if __name__ == "__main__":
    inference_example()
```

### 시각화 코드
```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_cross_attention(attention_weights, vision_patches=14, text_tokens=10):
    """Cross-modal attention 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Vision to Text attention
    v2t = attention_weights['vision_to_text'].detach().cpu().numpy()
    v2t = v2t.reshape(vision_patches, vision_patches, text_tokens)
    
    # 중앙 패치의 attention
    center_patch = v2t[7, 7, :]
    axes[0].bar(range(text_tokens), center_patch)
    axes[0].set_title('Center Patch Attention to Text Tokens')
    axes[0].set_xlabel('Text Token Index')
    axes[0].set_ylabel('Attention Weight')
    
    # Text to Vision attention heatmap
    t2v = attention_weights['text_to_vision'].detach().cpu().numpy()
    t2v = t2v.reshape(text_tokens, vision_patches * vision_patches)
    
    sns.heatmap(t2v[:5], ax=axes[1], cmap='YlOrRd')
    axes[1].set_title('Text Tokens Attention to Vision Patches')
    axes[1].set_xlabel('Vision Patch Index')
    axes[1].set_ylabel('Text Token Index')
    
    plt.tight_layout()
    plt.show()

def plot_modality_embeddings(vision_embeds, text_embeds):
    """모달리티 임베딩 분포 시각화"""
    from sklearn.manifold import TSNE
    
    # Combine embeddings
    all_embeds = torch.cat([vision_embeds, text_embeds], dim=0)
    labels = ['Vision'] * len(vision_embeds) + ['Text'] * len(text_embeds)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeds_2d = tsne.fit_transform(all_embeds.detach().cpu().numpy())
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = {'Vision': 'blue', 'Text': 'red'}
    for label in set(labels):
        mask = [l == label for l in labels]
        plt.scatter(embeds_2d[mask, 0], embeds_2d[mask, 1], 
                   label=label, alpha=0.6, c=colors[label])
    
    plt.legend()
    plt.title('Vision and Text Embeddings Distribution')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()
```

---

## 📈 다음 단계

### 1. 고급 멀티모달 기법
- **Mixture of Experts**: 모달리티별 전문가 모델
- **Adaptive Fusion**: 태스크에 따른 동적 융합
- **Multi-Scale Processing**: 다양한 해상도에서 처리

### 2. VLA 특화 기법
- **Temporal Multimodal**: 시간적 일관성 유지
- **Proprioceptive Integration**: 로봇 센서 데이터 통합
- **Multi-Camera Fusion**: 다중 카메라 입력 처리

### 3. 최신 연구 동향
- **Unified Transformers**: 모든 모달리티를 토큰으로
- **Diffusion Models**: 멀티모달 생성 모델
- **Neural Fields**: 3D 표현과 언어 결합

---

## 💡 핵심 포인트

### ✅ 기억해야 할 것들
1. **Early vs Late Fusion**: 태스크에 따라 선택
2. **Cross-Attention**: 모달리티 간 상호작용의 핵심
3. **Contrastive Learning**: 강력한 표현 학습 방법
4. **Alignment Quality**: 모달리티 정렬이 성능 좌우

### ⚠️ 주의사항
1. **Modality Imbalance**: 한 모달리티가 지배하지 않도록
2. **Computational Cost**: Cross-attention은 비용이 높음
3. **Data Requirements**: 페어링된 데이터 필요

### 🎯 VLA 적용 시
1. **Real-time Processing**: 효율적인 fusion 방법 선택
2. **Context Integration**: 이전 상호작용 기억
3. **Safety**: 모달리티 실패 시 fallback 메커니즘

---

**다음 문서**: `09_memory_context.md` - VLA의 메모리와 컨텍스트 관리