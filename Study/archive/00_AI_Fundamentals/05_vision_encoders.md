# 👁️ Vision Encoders: 로봇의 시각 처리

**목표**: CNN, Vision Transformer, 그리고 로봇 비전을 위한 최적화 기법 이해  
**시간**: 2-3시간  
**전제조건**: 01_neural_networks_basics.md, 03_transformer_architecture.md  

---

## 🎯 개발자를 위한 직관적 이해

### Vision Encoder의 역할
```python
vision_pipeline = {
    "input": "Raw pixel data (H×W×3)",
    "encoder": "Extract meaningful features",
    "output": "Compact representation (D-dim vector)"
}

# 로봇 비전의 특별한 요구사항
robot_vision_needs = {
    "real_time": "30+ FPS 처리 필요",
    "3d_understanding": "깊이, 거리 인식",
    "object_tracking": "움직이는 물체 추적",
    "robustness": "조명, 각도 변화에 강인"
}
```

### CNN vs ViT 비교
```python
comparison = {
    "CNN": {
        "장점": ["Inductive bias (locality)", "파라미터 효율적", "작은 데이터셋 OK"],
        "단점": ["Limited receptive field", "고정된 입력 크기"]
    },
    "ViT": {
        "장점": ["Global receptive field", "유연한 입력 크기", "Long-range dependencies"],
        "단점": ["많은 데이터 필요", "계산량 많음"]
    }
}
```

---

## 🏗️ 기본 구조 및 구현

### 1. CNN 기반 인코더 (ResNet)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """ResNet의 기본 빌딩 블록"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        out = F.relu(out)
        return out

class ResNetEncoder(nn.Module):
    """로봇 비전을 위한 ResNet 인코더"""
    def __init__(self, layers=[2, 2, 2, 2], num_classes=768):
        super().__init__()
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        # Global average pooling and projection
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def get_intermediate_features(self, x):
        """중간 특징 추출 (FPN 등에 활용)"""
        features = []
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append(x)  # 1/4 resolution
        
        x = self.layer2(x)
        features.append(x)  # 1/8 resolution
        
        x = self.layer3(x)
        features.append(x)  # 1/16 resolution
        
        x = self.layer4(x)
        features.append(x)  # 1/32 resolution
        
        return features
```

### 2. Vision Transformer (ViT)
```python
class PatchEmbedding(nn.Module):
    """이미지를 패치로 나누고 임베딩"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Convolution으로 패치 추출 및 임베딩
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)  # [B, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [B, n_patches, embed_dim]
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer 인코더"""
    def __init__(self, 
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.n_patches = self.patch_embed.n_patches
        
        # Positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Return CLS token as global representation
        return x[:, 0]
    
    def get_attention_maps(self, x):
        """Attention map 시각화용"""
        attention_maps = []
        
        # Initial processing
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        # Get attention from each block
        for block in self.blocks:
            x, attn = block(x, return_attention=True)
            attention_maps.append(attn)
        
        return attention_maps

class TransformerBlock(nn.Module):
    """ViT의 Transformer 블록"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, return_attention=False):
        # Self-attention
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        if return_attention:
            return x, attn_weights
        return x
```

### 3. EfficientNet (효율성 중시)
```python
class MBConvBlock(nn.Module):
    """MobileNet 스타일의 효율적인 convolution block"""
    def __init__(self, in_channels, out_channels, expand_ratio=4, kernel_size=3, stride=1):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        layers = []
        
        # Expansion phase
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.SiLU())
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ])
        
        # Squeeze and Excitation
        layers.append(SqueezeExcitation(hidden_dim))
        
        # Output phase
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out = out + x
        return out

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation module"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_channels = channels // reduction
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_channels, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)

class EfficientNetEncoder(nn.Module):
    """로봇을 위한 경량 EfficientNet"""
    def __init__(self, width_mult=1.0, depth_mult=1.0, output_dim=768):
        super().__init__()
        
        # Base configuration (EfficientNet-B0)
        base_widths = [32, 16, 24, 40, 80, 112, 192, 320]
        base_depths = [1, 2, 2, 3, 3, 4, 1]
        
        # Scale widths and depths
        widths = [int(w * width_mult) for w in base_widths]
        depths = [int(d * depth_mult) for d in base_depths]
        
        # Build network
        layers = []
        
        # Stem
        layers.append(nn.Conv2d(3, widths[0], 3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(widths[0]))
        layers.append(nn.SiLU())
        
        # MBConv blocks
        in_channels = widths[0]
        for i in range(len(depths)):
            out_channels = widths[i+1]
            for j in range(depths[i]):
                stride = 2 if j == 0 and i > 0 else 1
                layers.append(
                    MBConvBlock(in_channels, out_channels, 
                               expand_ratio=4, stride=stride)
                )
                in_channels = out_channels
        
        # Head
        layers.append(nn.Conv2d(in_channels, 1280, 1, bias=False))
        layers.append(nn.BatchNorm2d(1280))
        layers.append(nn.SiLU())
        layers.append(nn.AdaptiveAvgPool2d(1))
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(1280, output_dim)
    
    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
```

### 4. 로봇 비전 특화 인코더
```python
class RobotVisionEncoder(nn.Module):
    """로봇 비전에 최적화된 하이브리드 인코더"""
    def __init__(self, 
                 backbone='efficientnet',
                 use_depth=True,
                 use_temporal=True,
                 output_dim=768):
        super().__init__()
        
        self.use_depth = use_depth
        self.use_temporal = use_temporal
        
        # RGB encoder
        if backbone == 'efficientnet':
            self.rgb_encoder = EfficientNetEncoder(output_dim=output_dim//2)
        elif backbone == 'resnet':
            self.rgb_encoder = ResNetEncoder(num_classes=output_dim//2)
        else:
            self.rgb_encoder = VisionTransformer(embed_dim=output_dim//2)
        
        # Depth encoder (if using RGBD)
        if use_depth:
            self.depth_encoder = nn.Sequential(
                nn.Conv2d(1, 32, 7, stride=2, padding=3),
                nn.ReLU(),
                nn.Conv2d(32, 64, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, output_dim//2)
            )
        
        # Temporal encoder (for video)
        if use_temporal:
            self.temporal_encoder = nn.LSTM(
                output_dim, output_dim//2, 
                batch_first=True, bidirectional=True
            )
        
        # Feature fusion
        fusion_dim = output_dim
        if use_depth:
            fusion_dim += output_dim//2
        if use_temporal:
            fusion_dim = output_dim
            
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
        # Object-centric attention
        self.object_attention = ObjectCentricAttention(output_dim)
        
    def forward(self, rgb, depth=None, temporal_context=None):
        # Process RGB
        rgb_features = self.rgb_encoder(rgb)
        
        features = [rgb_features]
        
        # Process depth if available
        if self.use_depth and depth is not None:
            depth_features = self.depth_encoder(depth)
            features.append(depth_features)
        
        # Concatenate features
        combined = torch.cat(features, dim=-1)
        
        # Process temporal context if available
        if self.use_temporal and temporal_context is not None:
            # temporal_context: [batch, seq_len, feature_dim]
            temporal_out, _ = self.temporal_encoder(temporal_context)
            combined = temporal_out[:, -1, :]  # Take last output
        
        # Fusion
        fused = self.fusion(combined)
        
        # Object-centric refinement
        refined = self.object_attention(fused.unsqueeze(1))
        
        return refined.squeeze(1)

class ObjectCentricAttention(nn.Module):
    """물체 중심 attention 메커니즘"""
    def __init__(self, feature_dim, num_slots=7):
        super().__init__()
        self.num_slots = num_slots
        self.feature_dim = feature_dim
        
        # Slot initialization
        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, feature_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, num_slots, feature_dim))
        
        # Attention mechanism
        self.norm_input = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(feature_dim)
        
        self.attention = nn.MultiheadAttention(feature_dim, 4, batch_first=True)
        
        # Slot update
        self.gru = nn.GRUCell(feature_dim, feature_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
    def forward(self, inputs, num_iterations=3):
        batch_size = inputs.shape[0]
        
        # Initialize slots
        mu = self.slot_mu.expand(batch_size, -1, -1)
        sigma = self.slot_log_sigma.exp().expand(batch_size, -1, -1)
        slots = mu + sigma * torch.randn_like(mu)
        
        # Normalize inputs
        inputs = self.norm_input(inputs)
        
        # Iterative attention
        for _ in range(num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Attention: slots attend to inputs
            attn_out, _ = self.attention(slots, inputs, inputs)
            
            # GRU update
            slots = slots.reshape(-1, self.feature_dim)
            attn_out = attn_out.reshape(-1, self.feature_dim)
            slots = self.gru(attn_out, slots_prev.reshape(-1, self.feature_dim))
            slots = slots.reshape(batch_size, self.num_slots, self.feature_dim)
            
            # MLP update
            slots = slots + self.mlp(slots)
        
        # Aggregate slots
        return slots.mean(dim=1, keepdim=True)
```

---

## 🤖 VLA에서의 활용

### 1. Multi-Scale Vision for VLA
```python
class MultiScaleVisionVLA(nn.Module):
    """다양한 스케일의 시각 정보 처리"""
    def __init__(self):
        super().__init__()
        
        # Feature Pyramid Network
        self.backbone = ResNetEncoder()
        
        # FPN layers
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(256, 256, 1),
            nn.Conv2d(128, 256, 1),
            nn.Conv2d(64, 256, 1)
        ])
        
        # Scale-specific heads
        self.coarse_head = nn.Linear(256, 128)  # Navigation
        self.medium_head = nn.Linear(256, 128)  # Manipulation
        self.fine_head = nn.Linear(256, 128)    # Precise control
        
        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )
    
    def forward(self, image):
        # Get multi-scale features
        features = self.backbone.get_intermediate_features(image)
        
        # FPN processing
        fpn_features = []
        for i, (feature, conv) in enumerate(zip(features[::-1], self.fpn_convs)):
            fpn_feat = conv(feature)
            
            if i > 0:
                # Upsample and add
                prev_shape = fpn_features[-1].shape[-2:]
                fpn_feat = F.interpolate(fpn_feat, size=prev_shape, mode='bilinear')
                fpn_feat = fpn_feat + fpn_features[-1]
            
            fpn_features.append(fpn_feat)
        
        # Extract scale-specific information
        coarse = F.adaptive_avg_pool2d(fpn_features[0], 1).flatten(1)
        coarse = self.coarse_head(coarse)
        
        medium = F.adaptive_avg_pool2d(fpn_features[1], 1).flatten(1)
        medium = self.medium_head(medium)
        
        fine = F.adaptive_avg_pool2d(fpn_features[2], 1).flatten(1)
        fine = self.fine_head(fine)
        
        # Combine and generate action
        combined = torch.cat([coarse, medium, fine], dim=-1)
        action = self.action_decoder(combined)
        
        return action, {
            'coarse': coarse,
            'medium': medium,
            'fine': fine
        }
```

### 2. Efficient Real-time Vision
```python
class RealtimeVisionEncoder(nn.Module):
    """실시간 처리를 위한 경량 인코더"""
    def __init__(self, target_fps=30):
        super().__init__()
        
        # MobileNet-style backbone
        self.backbone = nn.Sequential(
            # Stem
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            
            # Depthwise separable convolutions
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 256, stride=2),
            
            # Global pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Lightweight projection
        self.projection = nn.Linear(256, 512)
        
        # Frame skipping for efficiency
        self.frame_skip = max(1, 30 // target_fps)
        self.frame_buffer = []
        
    def forward(self, x):
        # Frame skipping logic
        self.frame_buffer.append(x)
        
        if len(self.frame_buffer) < self.frame_skip:
            # Reuse previous features
            if hasattr(self, 'cached_features'):
                return self.cached_features
        else:
            # Process accumulated frames
            x = self.frame_buffer[-1]  # Use most recent
            self.frame_buffer = []
        
        # Efficient forward pass
        features = self.backbone(x)
        features = self.projection(features)
        
        # Cache for reuse
        self.cached_features = features
        
        return features

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, 
                                   stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6()
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
```

### 3. Vision-Language Alignment for VLA
```python
class VisionLanguageAlignedEncoder(nn.Module):
    """언어와 정렬된 비전 인코더"""
    def __init__(self, vision_dim=768, language_dim=768):
        super().__init__()
        
        # Vision backbone
        self.vision_encoder = VisionTransformer(embed_dim=vision_dim)
        
        # Region proposal network
        self.rpn = RegionProposalNetwork(vision_dim)
        
        # Object detector
        self.detector = nn.Sequential(
            nn.Linear(vision_dim, vision_dim // 2),
            nn.ReLU(),
            nn.Linear(vision_dim // 2, 100)  # 100 object classes
        )
        
        # Vision-language alignment
        self.vision_proj = nn.Linear(vision_dim, 512)
        self.language_proj = nn.Linear(language_dim, 512)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(512, 8, batch_first=True)
        
    def forward(self, image, language_features=None):
        # Extract vision features
        vision_features = self.vision_encoder(image)
        
        # Get object proposals
        proposals = self.rpn(vision_features)
        
        # Detect objects
        object_logits = self.detector(proposals)
        
        # Align with language if provided
        if language_features is not None:
            # Project to common space
            vision_proj = self.vision_proj(vision_features.unsqueeze(1))
            language_proj = self.language_proj(language_features)
            
            # Cross-modal attention
            aligned_vision, _ = self.cross_attention(
                vision_proj, language_proj, language_proj
            )
            
            return aligned_vision.squeeze(1), object_logits
        
        return vision_features, object_logits

class RegionProposalNetwork(nn.Module):
    """간단한 RPN 구현"""
    def __init__(self, feature_dim):
        super().__init__()
        self.conv = nn.Conv2d(feature_dim, 256, 3, padding=1)
        self.cls_head = nn.Conv2d(256, 2, 1)  # Object vs background
        self.reg_head = nn.Conv2d(256, 4, 1)  # Bounding box regression
        
    def forward(self, features):
        if features.dim() == 2:
            # Convert to spatial format
            batch_size = features.shape[0]
            features = features.unsqueeze(-1).unsqueeze(-1)
        
        x = F.relu(self.conv(features))
        cls_scores = self.cls_head(x)
        bbox_deltas = self.reg_head(x)
        
        # Simplified: return flattened features
        return x.mean(dim=[2, 3])
```

---

## 🔬 핵심 개념 정리

### 1. Receptive Field 분석
```python
def calculate_receptive_field(model, input_size=224):
    """모델의 receptive field 계산"""
    # Create gradient hook
    gradients = []
    
    def hook_fn(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    # Register hook on first conv layer
    first_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            break
    
    handle = first_conv.register_backward_hook(hook_fn)
    
    # Forward and backward pass
    input_tensor = torch.randn(1, 3, input_size, input_size, requires_grad=True)
    output = model(input_tensor)
    
    # Create point gradient at center
    grad = torch.zeros_like(output)
    grad[0, output.shape[1]//2] = 1
    output.backward(gradient=grad)
    
    # Analyze gradient spread
    input_grad = input_tensor.grad[0, 0].abs()
    receptive_field = (input_grad > 0.01).sum().item()
    
    handle.remove()
    
    return int(np.sqrt(receptive_field))
```

### 2. 모델 경량화 기법
```python
class ModelCompression:
    """모델 압축 기법들"""
    
    @staticmethod
    def prune_model(model, sparsity=0.5):
        """Weight pruning"""
        import torch.nn.utils.prune as prune
        
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=sparsity)
        
        return model
    
    @staticmethod
    def quantize_model(model):
        """INT8 quantization"""
        import torch.quantization as quant
        
        model.eval()
        model.qconfig = quant.get_default_qconfig('fbgemm')
        quant.prepare(model, inplace=True)
        quant.convert(model, inplace=True)
        
        return model
    
    @staticmethod
    def knowledge_distillation(student, teacher, dataloader, epochs=10):
        """Knowledge distillation"""
        criterion = nn.KLDivLoss()
        optimizer = torch.optim.Adam(student.parameters())
        
        teacher.eval()
        for epoch in range(epochs):
            for images, _ in dataloader:
                # Teacher prediction
                with torch.no_grad():
                    teacher_logits = teacher(images)
                
                # Student prediction
                student_logits = student(images)
                
                # Distillation loss
                loss = criterion(
                    F.log_softmax(student_logits / 3, dim=1),
                    F.softmax(teacher_logits / 3, dim=1)
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return student
```

---

## 🛠️ 실습 코드

### 완전한 로봇 비전 시스템
```python
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np

class RobotVisionSystem:
    """완전한 로봇 비전 시스템"""
    def __init__(self, device='cuda'):
        self.device = device
        
        # Load models
        self.rgb_encoder = EfficientNetEncoder().to(device)
        self.depth_estimator = self._load_depth_model()
        self.object_detector = self._load_detector()
        
        # Preprocessing
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Tracking
        self.tracker = ObjectTracker()
        
    def _load_depth_model(self):
        """깊이 추정 모델 로드"""
        # Simplified depth estimation network
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 7, stride=2, padding=3, output_padding=1),
            nn.Sigmoid()
        ).to(self.device)
    
    def _load_detector(self):
        """객체 검출 모델 로드"""
        # Simplified YOLO-style detector
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 85, 1)  # 80 classes + 4 bbox + 1 confidence
        ).to(self.device)
    
    def process_frame(self, frame):
        """단일 프레임 처리"""
        # Convert to PIL Image
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        
        # Preprocess
        input_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.rgb_encoder(input_tensor)
            depth = self.depth_estimator(input_tensor)
            detections = self.object_detector(input_tensor)
        
        # Parse detections
        objects = self._parse_detections(detections)
        
        # Update tracker
        tracked_objects = self.tracker.update(objects)
        
        return {
            'features': features,
            'depth': depth,
            'objects': tracked_objects
        }
    
    def _parse_detections(self, raw_detections):
        """검출 결과 파싱"""
        # Simplified parsing
        batch_size, channels, h, w = raw_detections.shape
        
        # Reshape to [batch, h*w, 85]
        detections = raw_detections.permute(0, 2, 3, 1).reshape(batch_size, -1, 85)
        
        # Get confidence scores
        confidences = torch.sigmoid(detections[..., 4])
        
        # Get class predictions
        class_probs = torch.softmax(detections[..., 5:], dim=-1)
        class_ids = class_probs.argmax(dim=-1)
        
        # Get bounding boxes
        bboxes = detections[..., :4]
        
        # Filter by confidence
        mask = confidences > 0.5
        
        objects = []
        for i in range(batch_size):
            valid_indices = mask[i].nonzero(as_tuple=True)[0]
            
            for idx in valid_indices:
                objects.append({
                    'bbox': bboxes[i, idx].cpu().numpy(),
                    'confidence': confidences[i, idx].item(),
                    'class_id': class_ids[i, idx].item()
                })
        
        return objects
    
    def process_video(self, video_path, skip_frames=1):
        """비디오 처리"""
        cap = cv2.VideoCapture(video_path)
        
        results = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % skip_frames == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                result = self.process_frame(frame_rgb)
                results.append(result)
            
            frame_count += 1
        
        cap.release()
        return results

class ObjectTracker:
    """간단한 객체 추적기"""
    def __init__(self, max_lost=10):
        self.max_lost = max_lost
        self.tracks = {}
        self.next_id = 0
        
    def update(self, detections):
        """추적 업데이트"""
        # Simplified tracking logic
        tracked = []
        
        for det in detections:
            # Assign ID (simplified - just incremental)
            det['track_id'] = self.next_id
            self.next_id += 1
            tracked.append(det)
        
        return tracked

# 사용 예시
def demo_robot_vision():
    # Initialize system
    vision_system = RobotVisionSystem(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Process single image
    image = Image.open('robot_view.jpg')
    result = vision_system.process_frame(image)
    
    print(f"Extracted features shape: {result['features'].shape}")
    print(f"Depth map shape: {result['depth'].shape}")
    print(f"Detected {len(result['objects'])} objects")
    
    # Process video
    video_results = vision_system.process_video('robot_task.mp4', skip_frames=5)
    print(f"Processed {len(video_results)} frames")
    
    # Visualize results
    visualize_results(image, result)

def visualize_results(image, result):
    """결과 시각화"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Depth map
    depth = result['depth'][0, 0].cpu().numpy()
    axes[1].imshow(depth, cmap='viridis')
    axes[1].set_title('Estimated Depth')
    axes[1].axis('off')
    
    # Object detection
    axes[2].imshow(image)
    for obj in result['objects']:
        bbox = obj['bbox']
        # Draw bounding box (simplified)
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], 
                            bbox[3]-bbox[1], fill=False, color='red', linewidth=2)
        axes[2].add_patch(rect)
    axes[2].set_title(f'Detected Objects ({len(result["objects"])})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test implementations
    print("Testing ResNet encoder...")
    resnet = ResNetEncoder()
    x = torch.randn(2, 3, 224, 224)
    out = resnet(x)
    print(f"ResNet output: {out.shape}")
    
    print("\nTesting ViT encoder...")
    vit = VisionTransformer()
    out = vit(x)
    print(f"ViT output: {out.shape}")
    
    print("\nTesting EfficientNet encoder...")
    efficientnet = EfficientNetEncoder()
    out = efficientnet(x)
    print(f"EfficientNet output: {out.shape}")
    
    print("\nTesting Robot Vision encoder...")
    robot_encoder = RobotVisionEncoder()
    rgb = torch.randn(2, 3, 224, 224)
    depth = torch.randn(2, 1, 224, 224)
    out = robot_encoder(rgb, depth)
    print(f"Robot encoder output: {out.shape}")
```

---

## 📈 다음 단계

### 1. 고급 비전 기법
- **Self-supervised Learning**: 라벨 없이 학습
- **Neural Architecture Search**: 자동 구조 탐색
- **Vision-Language Pre-training**: CLIP 스타일 학습

### 2. 로봇 특화 개선
- **Active Vision**: 능동적 시점 제어
- **3D Vision**: Point cloud 처리
- **Multi-camera Fusion**: 다중 카메라 통합

### 3. 최신 연구
- **Masked Autoencoders (MAE)**: 효율적 사전학습
- **Swin Transformer**: 계층적 비전 트랜스포머
- **ConvNeXt**: 현대화된 CNN

---

## 💡 핵심 포인트

### ✅ 기억해야 할 것들
1. **CNN vs ViT**: 각각의 장단점과 적용 상황
2. **효율성**: 실시간 처리를 위한 최적화
3. **Multi-scale**: 다양한 스케일 정보 활용
4. **Domain-specific**: 로봇 비전의 특수 요구사항

### ⚠️ 주의사항
1. **계산 비용**: 실시간 제약 고려
2. **Overfitting**: 작은 데이터셋 주의
3. **입력 전처리**: 정규화 중요

### 🎯 VLA 적용 시
1. **실시간성**: 30+ FPS 목표
2. **강건성**: 다양한 환경 조건
3. **해석가능성**: 어텐션 맵 활용

---

**다음 문서**: `06_language_models.md` - 자연어 명령 이해