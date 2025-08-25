# 🧠 Neural Networks Basics

**목표**: 개발자 관점에서 신경망의 기본 개념과 VLA에서의 활용을 이해

**시간**: 1-2시간

**전제조건**: Python, 기본적인 선형대수 (벡터, 행렬)

---

## 🎯 개발자를 위한 직관적 이해

### 신경망 = 복잡한 함수 근사기
```python
# 전통적인 프로그래밍
def traditional_function(x):
    if x > 5:
        return x * 2
    else:
        return x + 1

# 신경망 = 데이터로부터 학습하는 함수
def neural_network(x, weights, biases):
    # weights와 biases는 데이터로부터 학습됨
    return some_complex_computation(x, weights, biases)
```

### 핵심 아이디어
```python
neural_network_concept = {
    "입력": "숫자들의 벡터 (이미지, 텍스트, 센서 데이터 등)",
    "처리": "가중합 + 비선형 변환을 여러 층에서 반복",
    "출력": "원하는 형태의 예측값 (분류, 회귀, 액션 등)",
    "학습": "정답과 예측값의 차이를 줄이도록 가중치 업데이트"
}
```

---

## 🏗️ 기본 구조: 퍼셉트론부터 시작

### 1. 단순 퍼셉트론 (Linear Layer)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplePerceptron(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # y = W*x + b (가중합)
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

# 예시: 3차원 입력 → 1차원 출력
perceptron = SimplePerceptron(input_size=3, output_size=1)

# 입력 데이터
x = torch.tensor([1.0, 2.0, 3.0])  # 3차원 벡터
output = perceptron(x)  # 1차원 출력

print(f"입력: {x}")
print(f"출력: {output}")
print(f"가중치: {perceptron.linear.weight}")
print(f"편향: {perceptron.linear.bias}")
```

### 2. 비선형성 추가 (Activation Functions)
```python
class NonLinearPerceptron(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        # 비선형 활성화 함수 적용
        return torch.relu(self.linear(x))  # ReLU: max(0, x)

# 여러 활성화 함수 비교
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

activations = {
    "ReLU": F.relu(x),           # max(0, x)
    "Sigmoid": F.sigmoid(x),     # 1 / (1 + e^(-x))
    "Tanh": F.tanh(x),          # (e^x - e^(-x)) / (e^x + e^(-x))
    "GELU": F.gelu(x)           # x * Φ(x) (Gaussian Error Linear Unit)
}

for name, result in activations.items():
    print(f"{name:8}: {result}")
```

### 3. 다층 신경망 (Multi-Layer Perceptron)
```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # 여러 층을 쌓아서 복잡한 함수 표현
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        
        # 정규화 (선택적)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        
        # 드롭아웃 (과적합 방지)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # 첫 번째 층: 입력 → 은닉층
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 두 번째 층: 은닉층 → 은닉층
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 출력층: 은닉층 → 출력
        x = self.layer3(x)
        return x

# 예시: MNIST 손글씨 인식
mlp = MLP(input_size=784,    # 28x28 이미지
          hidden_size=256,   # 은닉층 크기
          output_size=10)    # 10개 클래스 (0~9 숫자)

# 더미 데이터로 테스트
batch_size = 32
dummy_image = torch.randn(batch_size, 784)  # 32개의 784차원 이미지
predictions = mlp(dummy_image)  # 32개의 10차원 예측값

print(f"입력 크기: {dummy_image.shape}")
print(f"출력 크기: {predictions.shape}")
```

---

## 📚 학습 과정: Forward & Backward Propagation

### 1. Forward Pass (순전파)
```python
def forward_pass_example():
    """
    Forward pass: 입력 → 예측값 계산
    """
    # 모델과 데이터 준비
    model = MLP(input_size=4, hidden_size=8, output_size=2)
    x = torch.randn(10, 4)  # 배치 크기 10, 특성 4개
    y_true = torch.randint(0, 2, (10,))  # 실제 레이블 (0 또는 1)
    
    # Forward pass
    y_pred = model(x)  # 모델 예측
    
    # 손실 계산
    loss = F.cross_entropy(y_pred, y_true)
    
    print(f"입력: {x.shape}")
    print(f"예측: {y_pred.shape}")
    print(f"실제: {y_true.shape}")
    print(f"손실: {loss.item():.4f}")
    
    return loss, model

forward_pass_example()
```

### 2. Backward Pass (역전파)
```python
def backward_pass_example():
    """
    Backward pass: 손실 → 그라디언트 계산 → 가중치 업데이트
    """
    # 모델, 데이터, 옵티마이저 준비
    model = MLP(input_size=4, hidden_size=8, output_size=2)
    x = torch.randn(10, 4)
    y_true = torch.randint(0, 2, (10,))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("=== 학습 전 ===")
    first_layer_weight = model.layer1.weight.clone()
    
    # 학습 루프
    for epoch in range(5):
        # Forward pass
        y_pred = model(x)
        loss = F.cross_entropy(y_pred, y_true)
        
        # Backward pass
        optimizer.zero_grad()  # 그라디언트 초기화
        loss.backward()        # 그라디언트 계산
        optimizer.step()       # 가중치 업데이트
        
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    print("=== 학습 후 ===")
    weight_change = (model.layer1.weight - first_layer_weight).abs().mean()
    print(f"가중치 변화량: {weight_change.item():.6f}")

backward_pass_example()
```

### 3. 그라디언트 계산 과정 이해
```python
def gradient_computation_example():
    """
    그라디언트가 어떻게 계산되는지 단계별로 확인
    """
    # 간단한 모델
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    w = torch.tensor([0.5, 0.8], requires_grad=True)
    b = torch.tensor(0.1, requires_grad=True)
    
    # Forward: y = sum(w * x) + b
    y = torch.sum(w * x) + b
    print(f"y = {y.item():.2f}")
    
    # Backward: dy/dx, dy/dw, dy/db 계산
    y.backward()
    
    print(f"x의 그라디언트: {x.grad}")  # dy/dx = w
    print(f"w의 그라디언트: {w.grad}")  # dy/dw = x
    print(f"b의 그라디언트: {b.grad}")  # dy/db = 1

gradient_computation_example()
```

---

## 🤖 VLA에서의 신경망 활용

### 1. Vision Encoder (이미지 → 특성 벡터)
```python
class VisionEncoder(nn.Module):
    """
    로봇 카메라 이미지를 특성 벡터로 변환
    """
    def __init__(self, image_size=224, feature_dim=512):
        super().__init__()
        # CNN 기반 특성 추출
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Global average pooling
        )
        
        self.fc = nn.Linear(256, feature_dim)
    
    def forward(self, images):
        # images: (batch, 3, 224, 224)
        features = self.conv_layers(images)  # (batch, 256, 1, 1)
        features = features.flatten(1)       # (batch, 256)
        features = self.fc(features)         # (batch, 512)
        return features

# 예시 사용
vision_encoder = VisionEncoder()
dummy_image = torch.randn(4, 3, 224, 224)  # 4개 이미지
visual_features = vision_encoder(dummy_image)
print(f"이미지 → 특성 벡터: {dummy_image.shape} → {visual_features.shape}")
```

### 2. Language Encoder (명령어 → 특성 벡터)
```python
class LanguageEncoder(nn.Module):
    """
    자연어 명령을 특성 벡터로 변환
    """
    def __init__(self, vocab_size=10000, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 512)
    
    def forward(self, tokens):
        # tokens: (batch, sequence_length)
        embedded = self.embedding(tokens)        # (batch, seq_len, embed_dim)
        _, (hidden, _) = self.lstm(embedded)     # hidden: (1, batch, hidden_dim)
        features = self.fc(hidden.squeeze(0))    # (batch, 512)
        return features

# 예시 사용
language_encoder = LanguageEncoder()
# "pick up the red cup" → [45, 123, 67, 891, 234]
dummy_tokens = torch.randint(0, 10000, (4, 8))  # 4개 문장, 각 8 토큰
language_features = language_encoder(dummy_tokens)
print(f"명령어 → 특성 벡터: {dummy_tokens.shape} → {language_features.shape}")
```

### 3. Action Decoder (특성 벡터 → 로봇 액션)
```python
class ActionDecoder(nn.Module):
    """
    통합된 특성을 로봇 액션으로 변환
    """
    def __init__(self, feature_dim=1024, action_dim=7):  # 7-DOF 로봇 팔
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # 액션 범위 제한 (예: -1 ~ 1)
        self.action_scale = nn.Parameter(torch.ones(action_dim))
    
    def forward(self, features):
        raw_actions = self.policy_head(features)
        # Tanh로 -1 ~ 1 범위로 제한
        normalized_actions = torch.tanh(raw_actions)
        # 스케일 적용
        actions = normalized_actions * self.action_scale
        return actions

# 통합 VLA 모델 예시
class SimpleVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = VisionEncoder()
        self.language_encoder = LanguageEncoder()
        self.action_decoder = ActionDecoder(feature_dim=1024)  # 512 + 512
    
    def forward(self, images, instructions):
        # 각 모달리티 인코딩
        visual_features = self.vision_encoder(images)      # (batch, 512)
        language_features = self.language_encoder(instructions)  # (batch, 512)
        
        # 특성 결합
        combined_features = torch.cat([visual_features, language_features], dim=1)  # (batch, 1024)
        
        # 액션 예측
        actions = self.action_decoder(combined_features)   # (batch, 7)
        
        return actions

# 전체 파이프라인 테스트
vla_model = SimpleVLA()
dummy_images = torch.randn(2, 3, 224, 224)
dummy_instructions = torch.randint(0, 10000, (2, 8))

predicted_actions = vla_model(dummy_images, dummy_instructions)
print(f"VLA 출력: {predicted_actions.shape}")  # (2, 7)
print(f"예측된 액션: {predicted_actions}")
```

---

## 🔬 핵심 개념 정리

### 1. 왜 신경망이 효과적인가?
```python
key_advantages = {
    "범용성": "어떤 함수든 근사 가능 (Universal Approximation Theorem)",
    "학습능력": "데이터로부터 자동으로 패턴 발견",
    "확장성": "더 많은 데이터와 계산으로 성능 향상",
    "미분가능성": "그라디언트 기반 최적화 가능"
}
```

### 2. VLA에서 신경망의 역할
```python
vla_neural_network_roles = {
    "특성추출": "원시 데이터(이미지, 텍스트)에서 의미있는 정보 추출",
    "모달리티융합": "서로 다른 종류의 정보(비전+언어) 결합",
    "정책학습": "상황에 맞는 최적의 행동 방법 학습",
    "표현학습": "고차원 데이터를 저차원으로 효과적 압축"
}
```

### 3. 주요 하이퍼파라미터
```python
important_hyperparameters = {
    "학습률": {
        "값": "0.001 ~ 0.1",
        "역할": "가중치 업데이트 보폭",
        "팁": "Adam 옵티마이저에서는 0.001이 좋은 시작점"
    },
    
    "배치크기": {
        "값": "16 ~ 128",
        "역할": "한 번에 처리하는 샘플 수",
        "팁": "GPU 메모리에 맞춰 조정"
    },
    
    "은닉층크기": {
        "값": "64 ~ 1024",
        "역할": "모델 표현력 결정",
        "팁": "너무 크면 과적합, 너무 작으면 성능 부족"
    },
    
    "드롭아웃": {
        "값": "0.1 ~ 0.5",
        "역할": "과적합 방지",
        "팁": "학습 중에만 적용, 추론 시에는 비활성화"
    }
}
```

---

## 🛠️ 실습: 간단한 VLA 모델 훈련

### 완전한 훈련 코드
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# 더미 VLA 데이터셋
class DummyVLADataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 더미 이미지 (3, 64, 64)
        image = torch.randn(3, 64, 64)
        
        # 더미 명령어 토큰 (길이 16)
        instruction = torch.randint(0, 1000, (16,))
        
        # 더미 액션 (7-DOF)
        action = torch.randn(7)
        
        return image, instruction, action

# 훈련 함수
def train_vla_model():
    # 데이터 준비
    dataset = DummyVLADataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 모델 (간단화된 버전)
    class SimpleVLAModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 간단한 CNN for vision
            self.vision = nn.Sequential(
                nn.Conv2d(3, 32, 3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(8),
                nn.Flatten(),
                nn.Linear(32*8*8, 256)
            )
            
            # 간단한 embedding for language
            self.language = nn.Sequential(
                nn.Embedding(1000, 64),
                nn.LSTM(64, 128, batch_first=True),
            )
            
            # Action decoder
            self.action_head = nn.Sequential(
                nn.Linear(256 + 128, 128),
                nn.ReLU(),
                nn.Linear(128, 7)
            )
            
        def forward(self, images, instructions):
            # Vision features
            vis_feat = self.vision(images)
            
            # Language features
            lang_embed = self.language[0](instructions)
            _, (lang_feat, _) = self.language[1](lang_embed)
            lang_feat = lang_feat.squeeze(0)
            
            # Combine and predict
            combined = torch.cat([vis_feat, lang_feat], dim=1)
            actions = self.action_head(combined)
            return actions
    
    model = SimpleVLAModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 훈련 루프
    losses = []
    for epoch in range(10):
        epoch_loss = 0
        for batch_idx, (images, instructions, target_actions) in enumerate(dataloader):
            # Forward pass
            predicted_actions = model(images, instructions)
            loss = criterion(predicted_actions, target_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    return model, losses

# 실행
if __name__ == "__main__":
    trained_model, training_losses = train_vla_model()
    print("훈련 완료!")
    print(f"최종 손실: {training_losses[-1]:.4f}")
```

---

## 📈 다음 단계

신경망 기초를 이해했다면:

1. **Attention Mechanism** (`02_attention_mechanism.md`) - VLA의 핵심
2. **Transformer Architecture** (`03_transformer_architecture.md`) - 현대 VLA의 기반
3. **Multi-Modal Learning** (`04_multimodal_learning.md`) - Vision + Language 결합

### 추천 실습
```python
recommended_practice = {
    "기초": "PyTorch 튜토리얼의 신경망 섹션 완주",
    "중급": "CIFAR-10 이미지 분류 모델 직접 구현",
    "고급": "간단한 로봇 제어 데이터셋으로 모방학습 시도"
}
```

---

## 💡 핵심 포인트

### 기억해야 할 것
1. **신경망 = 학습 가능한 함수**: 데이터로부터 입출력 관계 학습
2. **Forward + Backward**: 예측 → 손실 계산 → 그라디언트 → 가중치 업데이트
3. **비선형성 필수**: ReLU 같은 활성화 함수가 복잡한 패턴 학습 가능하게 함
4. **VLA = Vision + Language → Action**: 각 단계에서 신경망 활용

### VLA 연구에서의 의미
- **기본 빌딩 블록**: 모든 VLA 모델의 기초
- **확장 가능**: 더 복잡한 아키텍처(Transformer 등)의 구성 요소
- **해석 가능**: 각 층에서 무엇을 학습하는지 분석 가능

**다음: Attention 메커니즘으로!** 🚀

---

*Created: 2025-08-24*  
*Time: 1-2 hours*  
*Next: 02_attention_mechanism.md*