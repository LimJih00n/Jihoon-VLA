# 🔗 Cross-Modal Alignment: Vision과 Language의 정렬

**목표**: CLIP 구조, Contrastive Learning, Vision-Language 정렬 품질 측정 이해  
**시간**: 2-3시간  
**전제조건**: 01_neural_networks_basics.md, 02_attention_mechanism.md, 04_multimodal_learning.md  

---

## 🎯 개발자를 위한 직관적 이해

### Cross-Modal Alignment란?
```python
alignment_concept = {
    "goal": "서로 다른 모달리티를 같은 공간에 매핑",
    "vision": "이미지 → 벡터",
    "language": "텍스트 → 벡터",
    "alignment": "관련된 이미지-텍스트가 가까운 벡터"
}

# CLIP의 핵심 아이디어
clip_idea = {
    "positive_pairs": ("고양이 사진", "귀여운 고양이"),  # 가깝게
    "negative_pairs": ("고양이 사진", "자동차"),       # 멀게
    "learning": "대조 학습으로 정렬"
}

# VLA에서의 중요성
vla_benefits = {
    "zero_shot": "보지 못한 명령도 이해",
    "generalization": "새로운 객체/상황 처리",
    "reasoning": "시각-언어 추론 가능"
}
```

---

## 🏗️ 기본 구조 및 구현

### 1. CLIP (Contrastive Language-Image Pre-training)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CLIP(nn.Module):
    """CLIP 모델 구현"""
    def __init__(self, 
                 vision_encoder,
                 text_encoder,
                 embed_dim=512,
                 temperature=0.07):
        super().__init__()
        
        # Encoders
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        # Projection heads
        vision_output_dim = vision_encoder.output_dim
        text_output_dim = text_encoder.output_dim
        
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_output_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_output_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Temperature parameter for softmax
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        
    def encode_image(self, image):
        """이미지 인코딩"""
        vision_features = self.vision_encoder(image)
        vision_embed = self.vision_projection(vision_features)
        vision_embed = F.normalize(vision_embed, dim=-1)
        return vision_embed
    
    def encode_text(self, text):
        """텍스트 인코딩"""
        text_features = self.text_encoder(text)
        text_embed = self.text_projection(text_features)
        text_embed = F.normalize(text_embed, dim=-1)
        return text_embed
    
    def forward(self, images, texts):
        """Forward pass with contrastive loss"""
        # Encode both modalities
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(texts)
        
        # Compute similarity matrix
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text
    
    def compute_loss(self, logits_per_image, logits_per_text):
        """InfoNCE contrastive loss"""
        batch_size = logits_per_image.shape[0]
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=logits_per_image.device)
        
        # Cross entropy loss for both directions
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        
        total_loss = (loss_i2t + loss_t2i) / 2
        
        return total_loss
    
    def get_similarity(self, images, texts):
        """Get similarity scores between images and texts"""
        with torch.no_grad():
            image_embeds = self.encode_image(images)
            text_embeds = self.encode_text(texts)
            
            similarity = (image_embeds @ text_embeds.t()).cpu().numpy()
        
        return similarity

class CLIPTrainer:
    """CLIP 학습 관리"""
    def __init__(self, model, learning_rate=1e-4, warmup_steps=1000):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = self.create_scheduler(warmup_steps)
        
    def create_scheduler(self, warmup_steps):
        """Learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 1.0
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, images, texts):
        """Single training step"""
        self.model.train()
        
        # Forward pass
        logits_per_image, logits_per_text = self.model(images, texts)
        
        # Compute loss
        loss = self.model.compute_loss(logits_per_image, logits_per_text)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update
        self.optimizer.step()
        self.scheduler.step()
        
        # Compute accuracy
        with torch.no_grad():
            batch_size = images.shape[0]
            labels = torch.arange(batch_size, device=images.device)
            
            # Image to text accuracy
            i2t_acc = (logits_per_image.argmax(dim=1) == labels).float().mean()
            
            # Text to image accuracy
            t2i_acc = (logits_per_text.argmax(dim=1) == labels).float().mean()
        
        return {
            'loss': loss.item(),
            'i2t_acc': i2t_acc.item(),
            't2i_acc': t2i_acc.item()
        }
```

### 2. ALIGN (Alternative Alignment Method)
```python
class ALIGN(nn.Module):
    """ALIGN: Noisy data에 강건한 정렬"""
    def __init__(self, vision_encoder, text_encoder, embed_dim=512):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        # Projection with batch norm for stability
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_encoder.output_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_encoder.output_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        
        # Momentum encoders for stability
        self.momentum = 0.999
        self._build_momentum_encoder()
        
    def _build_momentum_encoder(self):
        """Build momentum encoder for more stable training"""
        # Copy encoders
        self.vision_encoder_m = copy.deepcopy(self.vision_encoder)
        self.text_encoder_m = copy.deepcopy(self.text_encoder)
        
        # No gradient for momentum encoders
        for param in self.vision_encoder_m.parameters():
            param.requires_grad = False
        for param in self.text_encoder_m.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update(self):
        """Update momentum encoders"""
        for param, param_m in zip(self.vision_encoder.parameters(), 
                                 self.vision_encoder_m.parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1 - self.momentum)
        
        for param, param_m in zip(self.text_encoder.parameters(),
                                 self.text_encoder_m.parameters()):
            param_m.data = param_m.data * self.momentum + param.data * (1 - self.momentum)
    
    def forward(self, images, texts, use_momentum=False):
        if use_momentum:
            # Use momentum encoders
            with torch.no_grad():
                vision_feat = self.vision_encoder_m(images)
                text_feat = self.text_encoder_m(texts)
        else:
            vision_feat = self.vision_encoder(images)
            text_feat = self.text_encoder(texts)
        
        # Project to common space
        vision_embed = self.vision_projection(vision_feat)
        text_embed = self.text_projection(text_feat)
        
        # L2 normalize
        vision_embed = F.normalize(vision_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        
        return vision_embed, text_embed
    
    def compute_loss(self, vision_embed, text_embed, temperature=0.07):
        """Normalized softmax loss"""
        batch_size = vision_embed.shape[0]
        
        # Similarity matrix
        sim_matrix = vision_embed @ text_embed.t() / temperature
        
        # Labels
        labels = torch.arange(batch_size, device=sim_matrix.device)
        
        # Softmax loss
        loss_v2t = F.cross_entropy(sim_matrix, labels)
        loss_t2v = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss_v2t + loss_t2v) / 2
```

### 3. Hierarchical Alignment
```python
class HierarchicalAlignment(nn.Module):
    """계층적 정렬: 다양한 수준에서 정렬"""
    def __init__(self, vision_encoder, text_encoder):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        # Multiple projection heads for different levels
        self.global_projection = ProjectionHead(512, 256)  # Global features
        self.local_projection = ProjectionHead(512, 256)   # Local features
        self.semantic_projection = ProjectionHead(512, 256) # Semantic features
        
        # Attention for local alignment
        self.cross_attention = nn.MultiheadAttention(256, 8, batch_first=True)
        
    def forward(self, images, texts):
        # Get multi-level features
        vision_features = self.vision_encoder.get_hierarchical_features(images)
        text_features = self.text_encoder.get_hierarchical_features(texts)
        
        # Global alignment
        global_v = self.global_projection(vision_features['global'])
        global_t = self.global_projection(text_features['global'])
        
        # Local alignment with attention
        local_v = self.local_projection(vision_features['local'])
        local_t = self.local_projection(text_features['local'])
        
        # Cross-modal attention for fine-grained alignment
        attended_v, v_weights = self.cross_attention(local_v, local_t, local_t)
        attended_t, t_weights = self.cross_attention(local_t, local_v, local_v)
        
        # Semantic alignment
        semantic_v = self.semantic_projection(vision_features['semantic'])
        semantic_t = self.semantic_projection(text_features['semantic'])
        
        return {
            'global': (global_v, global_t),
            'local': (attended_v, attended_t),
            'semantic': (semantic_v, semantic_t),
            'attention_weights': (v_weights, t_weights)
        }
    
    def compute_hierarchical_loss(self, outputs, weights=[1.0, 0.5, 0.5]):
        """Compute weighted loss across all levels"""
        total_loss = 0
        
        # Global loss
        global_v, global_t = outputs['global']
        global_loss = self.contrastive_loss(global_v, global_t)
        total_loss += weights[0] * global_loss
        
        # Local loss
        local_v, local_t = outputs['local']
        local_loss = self.contrastive_loss(local_v.mean(dim=1), local_t.mean(dim=1))
        total_loss += weights[1] * local_loss
        
        # Semantic loss
        semantic_v, semantic_t = outputs['semantic']
        semantic_loss = self.contrastive_loss(semantic_v, semantic_t)
        total_loss += weights[2] * semantic_loss
        
        return total_loss
    
    def contrastive_loss(self, embeds1, embeds2, temperature=0.07):
        """Standard contrastive loss"""
        embeds1 = F.normalize(embeds1, dim=-1)
        embeds2 = F.normalize(embeds2, dim=-1)
        
        logits = embeds1 @ embeds2.t() / temperature
        labels = torch.arange(len(logits), device=logits.device)
        
        return F.cross_entropy(logits, labels)

class ProjectionHead(nn.Module):
    """Projection head for alignment"""
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
```

### 4. Dynamic Alignment
```python
class DynamicAlignment(nn.Module):
    """동적 정렬: 태스크에 따라 정렬 방식 조정"""
    def __init__(self, vision_encoder, text_encoder, num_experts=4):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        # Mixture of experts for different alignment strategies
        self.experts = nn.ModuleList([
            AlignmentExpert(512, 256) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, images, texts, task_embedding=None):
        # Encode modalities
        vision_features = self.vision_encoder(images)
        text_features = self.text_encoder(texts)
        
        # Compute gating weights
        combined = torch.cat([vision_features, text_features], dim=-1)
        gate_weights = self.gate(combined)
        
        # Apply experts
        aligned_vision = torch.zeros_like(vision_features[:, :256])
        aligned_text = torch.zeros_like(text_features[:, :256])
        
        for i, expert in enumerate(self.experts):
            v_exp, t_exp = expert(vision_features, text_features)
            weight = gate_weights[:, i:i+1]
            aligned_vision += weight * v_exp
            aligned_text += weight * t_exp
        
        return aligned_vision, aligned_text, gate_weights

class AlignmentExpert(nn.Module):
    """Single alignment expert"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.vision_transform = nn.Linear(input_dim, output_dim)
        self.text_transform = nn.Linear(input_dim, output_dim)
        
    def forward(self, vision_features, text_features):
        v_aligned = self.vision_transform(vision_features)
        t_aligned = self.text_transform(text_features)
        return F.normalize(v_aligned, dim=-1), F.normalize(t_aligned, dim=-1)
```

---

## 🤖 VLA에서의 활용

### 1. Vision-Language-Action Alignment
```python
class VLAAlignment(nn.Module):
    """VLA를 위한 3-way alignment"""
    def __init__(self, vision_encoder, language_encoder, action_encoder):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.language_encoder = language_encoder
        self.action_encoder = action_encoder
        
        # Three-way projection
        embed_dim = 512
        self.vision_proj = nn.Linear(768, embed_dim)
        self.language_proj = nn.Linear(768, embed_dim)
        self.action_proj = nn.Linear(7, embed_dim)
        
        # Cross-modal transformers
        self.vl_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 8, batch_first=True),
            num_layers=2
        )
        self.va_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 8, batch_first=True),
            num_layers=2
        )
        
    def forward(self, images, instructions, actions):
        # Encode all modalities
        v_features = self.vision_encoder(images)
        l_features = self.language_encoder(instructions)
        a_features = self.action_encoder(actions)
        
        # Project to common space
        v_embed = self.vision_proj(v_features)
        l_embed = self.language_proj(l_features)
        a_embed = self.action_proj(a_features)
        
        # Vision-Language alignment
        vl_input = torch.stack([v_embed, l_embed], dim=1)
        vl_aligned = self.vl_transformer(vl_input)
        
        # Vision-Action alignment
        va_input = torch.stack([v_embed, a_embed], dim=1)
        va_aligned = self.va_transformer(va_input)
        
        return {
            'vision': v_embed,
            'language': l_embed,
            'action': a_embed,
            'vl_aligned': vl_aligned[:, 0],  # Take vision part
            'va_aligned': va_aligned[:, 0]   # Take vision part
        }
    
    def compute_triplet_loss(self, outputs, margin=0.2):
        """Triplet loss for three-way alignment"""
        v = F.normalize(outputs['vision'], dim=-1)
        l = F.normalize(outputs['language'], dim=-1)
        a = F.normalize(outputs['action'], dim=-1)
        
        # Positive pairs: (v,l), (v,a), (l,a) from same sample
        # Negative pairs: cross-sample combinations
        
        batch_size = v.shape[0]
        
        # Vision-Language triplet
        vl_pos = (v * l).sum(dim=-1)
        vl_neg = v @ l.t()
        vl_neg.fill_diagonal_(-1e10)  # Exclude positive pairs
        vl_neg = vl_neg.max(dim=-1)[0]
        vl_loss = F.relu(margin - vl_pos + vl_neg).mean()
        
        # Vision-Action triplet
        va_pos = (v * a).sum(dim=-1)
        va_neg = v @ a.t()
        va_neg.fill_diagonal_(-1e10)
        va_neg = va_neg.max(dim=-1)[0]
        va_loss = F.relu(margin - va_pos + va_neg).mean()
        
        # Language-Action triplet
        la_pos = (l * a).sum(dim=-1)
        la_neg = l @ a.t()
        la_neg.fill_diagonal_(-1e10)
        la_neg = la_neg.max(dim=-1)[0]
        la_loss = F.relu(margin - la_pos + la_neg).mean()
        
        return vl_loss + va_loss + la_loss
```

### 2. Grounded Language Understanding
```python
class GroundedAlignment(nn.Module):
    """객체 수준 grounded alignment"""
    def __init__(self, vision_encoder, language_encoder):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.language_encoder = language_encoder
        
        # Object detector
        self.object_detector = nn.Sequential(
            nn.Conv2d(768, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 100, 1)  # 100 object classes
        )
        
        # Word-level encoder
        self.word_encoder = nn.LSTM(768, 384, bidirectional=True, batch_first=True)
        
        # Grounding module
        self.grounding = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
    def forward(self, images, texts):
        # Visual features with spatial info
        visual_features = self.vision_encoder.get_spatial_features(images)
        
        # Detect objects
        object_logits = self.object_detector(visual_features)
        object_probs = F.softmax(object_logits, dim=1)
        
        # Word-level language features
        word_features, _ = self.word_encoder(texts)
        
        # Ground each word to visual regions
        grounded_features = []
        for word_feat in word_features.transpose(0, 1):
            # Compute attention over visual regions
            attention = self.compute_grounding_attention(
                word_feat, visual_features
            )
            
            # Weighted visual features
            weighted_visual = (visual_features * attention.unsqueeze(1)).sum(dim=(2, 3))
            
            # Combine word and visual
            combined = torch.cat([word_feat, weighted_visual], dim=-1)
            grounded = self.grounding(combined)
            grounded_features.append(grounded)
        
        grounded_features = torch.stack(grounded_features, dim=1)
        
        return {
            'grounded_features': grounded_features,
            'object_probs': object_probs,
            'word_features': word_features
        }
    
    def compute_grounding_attention(self, word_feat, visual_features):
        """Compute attention weights for grounding"""
        # word_feat: [batch, 768]
        # visual_features: [batch, 768, H, W]
        
        batch, channels, h, w = visual_features.shape
        
        # Reshape for attention computation
        visual_flat = visual_features.view(batch, channels, -1).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(visual_flat, word_feat.unsqueeze(-1)).squeeze(-1)
        attention = F.softmax(scores, dim=-1)
        
        # Reshape back to spatial
        attention = attention.view(batch, h, w)
        
        return attention
```

---

## 🔬 핵심 개념 정리

### 1. Alignment Quality Metrics
```python
def evaluate_alignment(model, test_loader):
    """정렬 품질 평가"""
    model.eval()
    
    metrics = {
        'retrieval_r1': 0,  # Recall@1
        'retrieval_r5': 0,  # Recall@5
        'retrieval_r10': 0, # Recall@10
        'mean_rank': 0,
        'semantic_similarity': 0,
        'modality_gap': 0
    }
    
    all_image_embeds = []
    all_text_embeds = []
    
    with torch.no_grad():
        for images, texts in test_loader:
            image_embeds = model.encode_image(images)
            text_embeds = model.encode_text(texts)
            
            all_image_embeds.append(image_embeds)
            all_text_embeds.append(text_embeds)
    
    # Concatenate all embeddings
    all_image_embeds = torch.cat(all_image_embeds)
    all_text_embeds = torch.cat(all_text_embeds)
    
    # Compute similarity matrix
    similarity = all_image_embeds @ all_text_embeds.t()
    
    # Image-to-text retrieval
    for i in range(len(similarity)):
        ranks = similarity[i].argsort(descending=True)
        rank = (ranks == i).nonzero()[0].item()
        
        metrics['mean_rank'] += rank
        if rank == 0:
            metrics['retrieval_r1'] += 1
        if rank < 5:
            metrics['retrieval_r5'] += 1
        if rank < 10:
            metrics['retrieval_r10'] += 1
    
    # Normalize metrics
    n = len(similarity)
    metrics['retrieval_r1'] /= n
    metrics['retrieval_r5'] /= n
    metrics['retrieval_r10'] /= n
    metrics['mean_rank'] /= n
    
    # Semantic similarity (diagonal elements)
    metrics['semantic_similarity'] = similarity.diagonal().mean().item()
    
    # Modality gap
    image_center = all_image_embeds.mean(dim=0)
    text_center = all_text_embeds.mean(dim=0)
    metrics['modality_gap'] = (image_center - text_center).norm().item()
    
    return metrics

def visualize_alignment(image_embeds, text_embeds, labels):
    """정렬 시각화"""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    # Combine embeddings
    all_embeds = torch.cat([image_embeds, text_embeds])
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeds_2d = tsne.fit_transform(all_embeds.cpu().numpy())
    
    # Split back
    n = len(image_embeds)
    image_2d = embeds_2d[:n]
    text_2d = embeds_2d[n:]
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Plot images
    plt.scatter(image_2d[:, 0], image_2d[:, 1], c='blue', label='Images', alpha=0.6)
    
    # Plot texts
    plt.scatter(text_2d[:, 0], text_2d[:, 1], c='red', label='Texts', alpha=0.6)
    
    # Draw lines between pairs
    for i in range(n):
        plt.plot([image_2d[i, 0], text_2d[i, 0]], 
                [image_2d[i, 1], text_2d[i, 1]], 
                'gray', alpha=0.3, linewidth=0.5)
    
    plt.legend()
    plt.title('Cross-Modal Alignment Visualization')
    plt.show()
```

### 2. Hard Negative Mining
```python
class HardNegativeMiner:
    """어려운 negative 샘플 선택"""
    def __init__(self, model, memory_bank_size=10000):
        self.model = model
        self.memory_bank = {
            'images': [],
            'texts': []
        }
        self.memory_bank_size = memory_bank_size
        
    def update_memory_bank(self, images, texts):
        """메모리 뱅크 업데이트"""
        with torch.no_grad():
            image_embeds = self.model.encode_image(images)
            text_embeds = self.model.encode_text(texts)
        
        self.memory_bank['images'].append(image_embeds)
        self.memory_bank['texts'].append(text_embeds)
        
        # Limit size
        if len(self.memory_bank['images']) > self.memory_bank_size:
            self.memory_bank['images'].pop(0)
            self.memory_bank['texts'].pop(0)
    
    def mine_hard_negatives(self, anchor_embeds, modality='image', k=10):
        """Find k hardest negatives"""
        if modality == 'image':
            negative_embeds = torch.cat(self.memory_bank['texts'])
        else:
            negative_embeds = torch.cat(self.memory_bank['images'])
        
        # Compute similarities
        similarities = anchor_embeds @ negative_embeds.t()
        
        # Get top-k hard negatives (highest similarity but wrong pairs)
        hard_negative_indices = similarities.topk(k, dim=1)[1]
        
        hard_negatives = negative_embeds[hard_negative_indices]
        
        return hard_negatives
```

---

## 🛠️ 실습 코드

### 완전한 Cross-Modal VLA 시스템
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CrossModalVLA:
    """완전한 Cross-Modal VLA 시스템"""
    def __init__(self, device='cuda'):
        self.device = device
        
        # Initialize encoders
        self.vision_encoder = self._build_vision_encoder()
        self.language_encoder = self._build_language_encoder()
        
        # Initialize CLIP model
        self.clip = CLIP(
            self.vision_encoder,
            self.language_encoder,
            embed_dim=512
        ).to(device)
        
        # Initialize VLA components
        self.vla_alignment = VLAAlignment(
            self.vision_encoder,
            self.language_encoder,
            self._build_action_encoder()
        ).to(device)
        
        # Training components
        self.optimizer = torch.optim.AdamW(
            list(self.clip.parameters()) + list(self.vla_alignment.parameters()),
            lr=1e-4
        )
        
        self.hard_negative_miner = HardNegativeMiner(self.clip)
        
    def _build_vision_encoder(self):
        """Build vision encoder"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 768)
        )
    
    def _build_language_encoder(self):
        """Build language encoder"""
        return nn.Sequential(
            nn.Embedding(10000, 768),
            nn.LSTM(768, 384, bidirectional=True, batch_first=True),
            Lambda(lambda x: x[0][:, -1, :])  # Take last output
        )
    
    def _build_action_encoder(self):
        """Build action encoder"""
        return nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 768)
        )
    
    def train_alignment(self, dataloader, epochs=10):
        """Train cross-modal alignment"""
        for epoch in range(epochs):
            total_loss = 0
            total_acc = 0
            
            for batch in dataloader:
                images = batch['image'].to(self.device)
                texts = batch['text'].to(self.device)
                actions = batch['action'].to(self.device)
                
                # CLIP alignment
                logits_i2t, logits_t2i = self.clip(images, texts)
                clip_loss = self.clip.compute_loss(logits_i2t, logits_t2i)
                
                # VLA alignment
                vla_outputs = self.vla_alignment(images, texts, actions)
                vla_loss = self.vla_alignment.compute_triplet_loss(vla_outputs)
                
                # Hard negative mining
                if epoch > 5:  # Start after some warmup
                    with torch.no_grad():
                        image_embeds = self.clip.encode_image(images)
                    
                    hard_negatives = self.hard_negative_miner.mine_hard_negatives(
                        image_embeds, modality='image', k=5
                    )
                    
                    # Additional loss with hard negatives
                    hard_neg_loss = self.compute_hard_negative_loss(
                        image_embeds, hard_negatives
                    )
                    
                    total_loss = clip_loss + vla_loss + 0.1 * hard_neg_loss
                else:
                    total_loss = clip_loss + vla_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.clip.parameters(), 1.0)
                self.optimizer.step()
                
                # Update memory bank
                self.hard_negative_miner.update_memory_bank(images, texts)
                
                # Compute accuracy
                with torch.no_grad():
                    i2t_pred = logits_i2t.argmax(dim=1)
                    i2t_acc = (i2t_pred == torch.arange(len(images)).to(self.device)).float().mean()
                    total_acc += i2t_acc.item()
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Loss: {total_loss:.4f}")
            print(f"  Accuracy: {total_acc/len(dataloader):.2%}")
    
    def compute_hard_negative_loss(self, anchors, hard_negatives, margin=0.2):
        """Compute loss with hard negatives"""
        # anchors: [batch, embed_dim]
        # hard_negatives: [batch, k, embed_dim]
        
        batch_size = anchors.shape[0]
        k = hard_negatives.shape[1]
        
        # Expand anchors
        anchors_exp = anchors.unsqueeze(1).expand(-1, k, -1)
        
        # Compute distances
        distances = F.cosine_similarity(anchors_exp, hard_negatives, dim=-1)
        
        # Hinge loss
        loss = F.relu(margin - distances).mean()
        
        return loss
    
    def zero_shot_classification(self, images, text_prompts):
        """Zero-shot classification using alignment"""
        self.clip.eval()
        
        with torch.no_grad():
            # Encode images
            image_embeds = self.clip.encode_image(images)
            
            # Encode all text prompts
            text_embeds = []
            for prompt in text_prompts:
                text_embed = self.clip.encode_text(prompt)
                text_embeds.append(text_embed)
            
            text_embeds = torch.stack(text_embeds)
            
            # Compute similarities
            similarities = image_embeds @ text_embeds.t()
            
            # Get predictions
            predictions = similarities.argmax(dim=1)
        
        return predictions
    
    def generate_action_from_alignment(self, image, instruction):
        """Generate robot action using cross-modal alignment"""
        self.vla_alignment.eval()
        
        with torch.no_grad():
            # Dummy action for alignment computation
            dummy_action = torch.zeros(1, 7).to(self.device)
            
            # Get aligned features
            outputs = self.vla_alignment(
                image.unsqueeze(0),
                instruction.unsqueeze(0),
                dummy_action
            )
            
            # Use aligned features to generate action
            aligned_features = outputs['vl_aligned']
            
            # Simple action decoder (would be more complex in practice)
            action = torch.tanh(aligned_features[:, :7])
        
        return action

class Lambda(nn.Module):
    """Lambda layer for functional operations"""
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def forward(self, x):
        return self.func(x)

# Demo dataset
class VLADataset(Dataset):
    """Demo dataset for VLA training"""
    def __init__(self, size=1000):
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'image': torch.randn(3, 224, 224),
            'text': torch.randint(0, 10000, (20,)),
            'action': torch.randn(7)
        }

# Evaluation utilities
def evaluate_zero_shot(model, test_images, test_labels, class_names):
    """Evaluate zero-shot performance"""
    # Create text prompts
    text_prompts = [f"a photo of a {name}" for name in class_names]
    
    # Get predictions
    predictions = model.zero_shot_classification(test_images, text_prompts)
    
    # Compute accuracy
    accuracy = (predictions == test_labels).float().mean()
    
    # Per-class accuracy
    per_class_acc = {}
    for i, name in enumerate(class_names):
        mask = test_labels == i
        if mask.any():
            class_acc = (predictions[mask] == i).float().mean()
            per_class_acc[name] = class_acc.item()
    
    return {
        'overall_accuracy': accuracy.item(),
        'per_class_accuracy': per_class_acc
    }

def visualize_retrieval(model, query_image, text_database, top_k=5):
    """Visualize image-to-text retrieval"""
    import matplotlib.pyplot as plt
    
    model.eval()
    
    with torch.no_grad():
        # Encode query image
        image_embed = model.encode_image(query_image.unsqueeze(0))
        
        # Encode all texts
        text_embeds = []
        for text in text_database:
            text_embed = model.encode_text(text)
            text_embeds.append(text_embed)
        
        text_embeds = torch.cat(text_embeds)
        
        # Compute similarities
        similarities = (image_embed @ text_embeds.t()).squeeze()
        
        # Get top-k
        top_k_values, top_k_indices = similarities.topk(top_k)
    
    # Visualize
    fig, axes = plt.subplots(1, top_k + 1, figsize=(3 * (top_k + 1), 3))
    
    # Show query image
    axes[0].imshow(query_image.permute(1, 2, 0))
    axes[0].set_title('Query Image')
    axes[0].axis('off')
    
    # Show retrieved texts
    for i, (idx, score) in enumerate(zip(top_k_indices, top_k_values)):
        axes[i + 1].text(0.5, 0.5, text_database[idx], 
                        ha='center', va='center', fontsize=10)
        axes[i + 1].set_title(f'Score: {score:.3f}')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Main demo
if __name__ == "__main__":
    # Initialize system
    system = CrossModalVLA(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    dataset = VLADataset(size=1000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train alignment
    system.train_alignment(dataloader, epochs=10)
    
    # Test zero-shot classification
    test_images = torch.randn(100, 3, 224, 224)
    class_names = ['robot', 'arm', 'gripper', 'object', 'table']
    test_labels = torch.randint(0, len(class_names), (100,))
    
    results = evaluate_zero_shot(system.clip, test_images, test_labels, class_names)
    print(f"Zero-shot accuracy: {results['overall_accuracy']:.2%}")
    
    # Test action generation
    test_image = torch.randn(3, 224, 224)
    test_instruction = torch.randint(0, 10000, (20,))
    action = system.generate_action_from_alignment(test_image, test_instruction)
    print(f"Generated action: {action.squeeze()}")
```

---

## 📈 다음 단계

### 1. 고급 정렬 기법
- **ALBEF**: Momentum distillation
- **BLIP**: Bootstrapping for better alignment
- **CoCa**: Contrastive captioners

### 2. VLA 특화 개선
- **Temporal alignment**: 시간적 일관성
- **Multi-task alignment**: 다중 작업 정렬
- **Embodied alignment**: 신체화된 정렬

### 3. 최신 연구
- **LLaVA**: Large language and vision assistant
- **Flamingo**: Few-shot vision-language
- **DALL-E**: Generation through alignment

---

## 💡 핵심 포인트

### ✅ 기억해야 할 것들
1. **Contrastive learning**: 정렬의 핵심
2. **Temperature parameter**: 학습 안정성
3. **Hard negatives**: 더 강한 정렬
4. **Multi-level alignment**: 다양한 수준

### ⚠️ 주의사항
1. **Modality gap**: 모달리티 간 거리
2. **False negatives**: 실제 매칭 쌍 처리
3. **Batch size**: 대조 학습에 중요

### 🎯 VLA 적용 시
1. **Three-way alignment**: Vision-Language-Action
2. **Grounded understanding**: 객체 수준 정렬
3. **Zero-shot transfer**: 새로운 태스크 일반화

---

**다음 문서**: `07_reinforcement_learning.md` - 강화학습 기초