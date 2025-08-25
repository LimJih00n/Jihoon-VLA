# 🌊 Flow Models: 연속적인 변환을 통한 생성

**목표**: Normalizing Flow, Flow Matching, 그리고 π₀ 스타일 Action Generation 이해  
**시간**: 3-4시간  
**전제조건**: 01_neural_networks_basics.md, 03_transformer_architecture.md  

---

## 🎯 개발자를 위한 직관적 이해

### Flow Model이란?
```python
flow_concept = {
    "idea": "단순한 분포 → 복잡한 분포로 변환",
    "simple": "가우시안 노이즈 z ~ N(0, I)",
    "complex": "실제 데이터 분포 x",
    "transform": "가역적(invertible) 변환 f",
    "equation": "x = f(z), z = f^(-1)(x)"
}

# 왜 Flow Models인가?
flow_advantages = {
    "exact_likelihood": "정확한 확률 계산 가능",
    "bidirectional": "생성과 인코딩 둘 다 가능",
    "continuous": "연속적인 변환 (불연속 없음)",
    "stable": "안정적인 학습"
}

# VLA에서의 활용 (π₀)
vla_flow = {
    "input": "비전-언어 특징",
    "output": "로봇 액션 (연속적)",
    "advantage": "부드러운 행동 생성"
}
```

---

## 🏗️ 기본 구조 및 구현

### 1. Normalizing Flow 기초
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NormalizingFlow(nn.Module):
    """기본 Normalizing Flow 구현"""
    def __init__(self, dim, n_flows=10):
        super().__init__()
        self.dim = dim
        self.flows = nn.ModuleList([
            PlanarFlow(dim) for _ in range(n_flows)
        ])
        
    def forward(self, z):
        """Forward transformation: z → x"""
        log_det_sum = 0
        
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_sum += log_det
        
        return z, log_det_sum
    
    def inverse(self, x):
        """Inverse transformation: x → z"""
        log_det_sum = 0
        
        for flow in reversed(self.flows):
            x, log_det = flow.inverse(x)
            log_det_sum += log_det
        
        return x, log_det_sum
    
    def log_likelihood(self, x):
        """Compute log-likelihood of data"""
        # Transform to base distribution
        z, log_det = self.inverse(x)
        
        # Log probability in base distribution
        log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * self.dim * np.log(2 * np.pi)
        
        # Change of variables formula
        log_px = log_pz + log_det
        
        return log_px
    
    def sample(self, n_samples):
        """Generate samples"""
        # Sample from base distribution
        z = torch.randn(n_samples, self.dim)
        
        # Transform to data distribution
        x, _ = self.forward(z)
        
        return x

class PlanarFlow(nn.Module):
    """Planar flow transformation"""
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim))
        self.bias = nn.Parameter(torch.randn(1))
        self.scale = nn.Parameter(torch.randn(dim))
        
    def forward(self, z):
        # Compute transformation
        activation = F.tanh(torch.matmul(z, self.weight) + self.bias)
        z_new = z + self.scale * activation.unsqueeze(-1)
        
        # Compute log determinant
        psi = (1 - activation ** 2) * self.weight
        log_det = torch.log(torch.abs(1 + torch.matmul(psi, self.scale)) + 1e-8)
        
        return z_new, log_det
    
    def inverse(self, x):
        """Inverse is not analytically tractable for planar flow"""
        # Use iterative solver or approximation
        raise NotImplementedError("Planar flow inverse requires numerical solution")
```

### 2. RealNVP (Real-valued Non-Volume Preserving)
```python
class RealNVP(nn.Module):
    """RealNVP: tractable normalizing flow"""
    def __init__(self, dim, n_flows=8, hidden_dim=256):
        super().__init__()
        self.dim = dim
        self.flows = nn.ModuleList()
        
        for i in range(n_flows):
            # Alternate masks for coupling layers
            mask = self.create_mask(dim, i % 2 == 0)
            self.flows.append(CouplingLayer(dim, mask, hidden_dim))
        
    def create_mask(self, dim, even):
        """Create checkerboard or channel-wise mask"""
        mask = torch.zeros(dim)
        if even:
            mask[::2] = 1
        else:
            mask[1::2] = 1
        return mask
    
    def forward(self, z, reverse=False):
        log_det_sum = 0
        
        flows = reversed(self.flows) if reverse else self.flows
        
        for flow in flows:
            z, log_det = flow(z, reverse=reverse)
            log_det_sum += log_det
        
        return z, log_det_sum
    
    def log_likelihood(self, x):
        z, log_det = self.forward(x, reverse=True)
        log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * self.dim * np.log(2 * np.pi)
        return log_pz + log_det
    
    def sample(self, n_samples):
        z = torch.randn(n_samples, self.dim)
        x, _ = self.forward(z, reverse=False)
        return x

class CouplingLayer(nn.Module):
    """Affine coupling layer"""
    def __init__(self, dim, mask, hidden_dim=256):
        super().__init__()
        self.register_buffer('mask', mask)
        
        # Scale and translation networks
        self.scale_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        
        self.translate_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x, reverse=False):
        # Split input
        x_masked = x * self.mask
        
        # Compute scale and translation
        log_scale = self.scale_net(x_masked)
        translation = self.translate_net(x_masked)
        
        # Apply only to non-masked parts
        log_scale = log_scale * (1 - self.mask)
        translation = translation * (1 - self.mask)
        
        if not reverse:
            # Forward transformation
            y = x * torch.exp(log_scale) + translation
            log_det = log_scale.sum(dim=-1)
        else:
            # Inverse transformation
            y = (x - translation) * torch.exp(-log_scale)
            log_det = -log_scale.sum(dim=-1)
        
        return y, log_det
```

### 3. Continuous Normalizing Flow (Neural ODE)
```python
class ContinuousNormalizingFlow(nn.Module):
    """Continuous normalizing flow using Neural ODE"""
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        
        # ODE function
        self.ode_func = ODEFunc(dim, hidden_dim)
        
        # Integration times
        self.register_buffer('integration_times', torch.tensor([0., 1.]))
        
    def forward(self, z):
        """Integrate ODE forward"""
        # Augment with log-determinant
        aug_z = torch.cat([z, torch.zeros(z.shape[0], 1)], dim=-1)
        
        # Solve ODE
        aug_x = odeint(
            self.ode_func,
            aug_z,
            self.integration_times,
            method='dopri5'
        )[-1]
        
        # Split output
        x = aug_x[:, :-1]
        log_det = aug_x[:, -1]
        
        return x, log_det
    
    def inverse(self, x):
        """Integrate ODE backward"""
        aug_x = torch.cat([x, torch.zeros(x.shape[0], 1)], dim=-1)
        
        # Solve ODE backward
        aug_z = odeint(
            self.ode_func,
            aug_x,
            torch.flip(self.integration_times, dims=[0]),
            method='dopri5'
        )[-1]
        
        z = aug_z[:, :-1]
        log_det = -aug_z[:, -1]  # Negative for inverse
        
        return z, log_det

class ODEFunc(nn.Module):
    """ODE function for continuous flow"""
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, t, aug_state):
        """Compute dynamics and trace"""
        state = aug_state[:, :-1]
        
        # Add time to input
        t_vec = torch.ones(state.shape[0], 1) * t
        input_vec = torch.cat([state, t_vec], dim=-1)
        
        # Compute dynamics
        dynamics = self.net(input_vec)
        
        # Compute trace for log-determinant
        trace = self.compute_trace(dynamics, state)
        
        # Augmented dynamics
        aug_dynamics = torch.cat([dynamics, -trace.unsqueeze(-1)], dim=-1)
        
        return aug_dynamics
    
    def compute_trace(self, dynamics, state):
        """Compute trace of Jacobian (Hutchinson estimator)"""
        # Random vector for trace estimation
        eps = torch.randn_like(state)
        
        # Compute Jacobian-vector product
        jvp = torch.autograd.grad(
            dynamics, state,
            grad_outputs=eps,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Estimate trace
        trace = (jvp * eps).sum(dim=-1)
        
        return trace

# Simple ODE solver (would use torchdiffeq in practice)
def odeint(func, y0, t, method='euler'):
    """Simple ODE integration"""
    trajectory = [y0]
    y = y0
    
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        
        if method == 'euler':
            dy = func(t[i], y)
            y = y + dt * dy
        elif method == 'dopri5':
            # Simplified RK45 (would use adaptive step size)
            k1 = func(t[i], y)
            k2 = func(t[i] + dt/2, y + dt*k1/2)
            k3 = func(t[i] + dt/2, y + dt*k2/2)
            k4 = func(t[i] + dt, y + dt*k3)
            y = y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        trajectory.append(y)
    
    return torch.stack(trajectory)
```

### 4. Flow Matching (π₀ Style)
```python
class FlowMatching(nn.Module):
    """Flow Matching for action generation (π₀ style)"""
    def __init__(self, condition_dim, action_dim, time_dim=128):
        super().__init__()
        self.condition_dim = condition_dim
        self.action_dim = action_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # Velocity network (predicts flow field)
        self.velocity_net = nn.Sequential(
            nn.Linear(action_dim + condition_dim + time_dim, 512),
            nn.SiLU(),
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Linear(512, action_dim)
        )
        
    def forward(self, x_t, t, condition):
        """Predict velocity field v(x_t, t)"""
        # Embed time
        t_embed = self.time_embed(t)
        
        # Concatenate inputs
        input_vec = torch.cat([x_t, condition, t_embed], dim=-1)
        
        # Predict velocity
        velocity = self.velocity_net(input_vec)
        
        return velocity
    
    def compute_loss(self, x_0, x_1, condition):
        """Compute flow matching loss"""
        batch_size = x_0.shape[0]
        
        # Sample time uniformly
        t = torch.rand(batch_size, 1).to(x_0.device)
        
        # Interpolate between x_0 and x_1
        x_t = (1 - t) * x_0 + t * x_1
        
        # Target velocity (straight path)
        v_target = x_1 - x_0
        
        # Predicted velocity
        v_pred = self.forward(x_t, t, condition)
        
        # MSE loss
        loss = F.mse_loss(v_pred, v_target)
        
        return loss
    
    @torch.no_grad()
    def generate(self, condition, n_steps=100):
        """Generate actions using ODE integration"""
        batch_size = condition.shape[0]
        device = condition.device
        
        # Start from noise
        x = torch.randn(batch_size, self.action_dim).to(device)
        
        # Time steps
        dt = 1.0 / n_steps
        
        for i in range(n_steps):
            t = torch.ones(batch_size, 1).to(device) * (i * dt)
            
            # Predict velocity
            v = self.forward(x, t, condition)
            
            # Euler integration
            x = x + v * dt
        
        return x

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """Residual block for deep networks"""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x):
        return x + self.block(x)
```

---

## 🤖 VLA에서의 활용

### 1. Flow-based Action Generation for VLA
```python
class FlowVLA(nn.Module):
    """Flow-based VLA for smooth action generation"""
    def __init__(self, vision_encoder, language_encoder, action_dim=7):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.language_encoder = language_encoder
        
        # Cross-modal fusion
        self.fusion = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Flow model for actions
        self.action_flow = FlowMatching(
            condition_dim=256,
            action_dim=action_dim
        )
        
        # Optional: Action refinement network
        self.refiner = nn.Sequential(
            nn.Linear(action_dim + 256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, image, instruction):
        """Generate action using flow"""
        # Encode inputs
        vision_features = self.vision_encoder(image)
        language_features = self.language_encoder(instruction)
        
        # Fuse modalities
        combined = torch.cat([vision_features, language_features], dim=-1)
        condition = self.fusion(combined)
        
        # Generate action with flow
        action = self.action_flow.generate(condition)
        
        # Optional refinement
        refined_input = torch.cat([action, condition], dim=-1)
        action = action + self.refiner(refined_input)
        
        return action
    
    def train_step(self, batch):
        """Training step with flow matching"""
        images = batch['images']
        instructions = batch['instructions']
        expert_actions = batch['actions']
        
        # Get conditions
        vision_features = self.vision_encoder(images)
        language_features = self.language_encoder(instructions)
        combined = torch.cat([vision_features, language_features], dim=-1)
        condition = self.fusion(combined)
        
        # Sample noise as x_0
        x_0 = torch.randn_like(expert_actions)
        
        # Expert actions as x_1
        x_1 = expert_actions
        
        # Compute flow matching loss
        loss = self.action_flow.compute_loss(x_0, x_1, condition)
        
        return loss

class DiffusionPolicy(nn.Module):
    """Diffusion-based policy (alternative to flow)"""
    def __init__(self, state_dim, action_dim, n_timesteps=100):
        super().__init__()
        
        self.n_timesteps = n_timesteps
        
        # Noise schedule
        self.betas = torch.linspace(0.0001, 0.02, n_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Denoising network
        self.denoiser = nn.Sequential(
            nn.Linear(action_dim + state_dim + 1, 256),  # +1 for timestep
            nn.ReLU(),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, x_t, t, state):
        """Predict noise ε(x_t, t, state)"""
        t_embed = t.float() / self.n_timesteps
        input_vec = torch.cat([x_t, state, t_embed.unsqueeze(-1)], dim=-1)
        noise_pred = self.denoiser(input_vec)
        return noise_pred
    
    def q_sample(self, x_0, t):
        """Forward diffusion process"""
        noise = torch.randn_like(x_0)
        alpha_t = self.alphas_cumprod[t]
        
        x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
        
        return x_t, noise
    
    @torch.no_grad()
    def sample(self, state, n_samples=1):
        """Reverse diffusion to generate actions"""
        # Start from noise
        x = torch.randn(n_samples, self.denoiser[-1].out_features)
        
        for t in reversed(range(self.n_timesteps)):
            t_batch = torch.full((n_samples,), t, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.forward(x, t_batch, state)
            
            # Denoise step
            alpha_t = self.alphas[t]
            alpha_t_cumprod = self.alphas_cumprod[t]
            
            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(self.betas[t])
            else:
                noise = 0
                sigma = 0
            
            x = (1 / torch.sqrt(alpha_t)) * (
                x - (1 - alpha_t) / torch.sqrt(1 - alpha_t_cumprod) * noise_pred
            ) + sigma * noise
        
        return x
```

### 2. Hierarchical Flow for Complex Actions
```python
class HierarchicalFlow(nn.Module):
    """Hierarchical flow for complex action sequences"""
    def __init__(self, condition_dim, action_dim, sequence_length=10):
        super().__init__()
        
        self.sequence_length = sequence_length
        
        # High-level flow (generates action sequence structure)
        self.high_level_flow = FlowMatching(
            condition_dim=condition_dim,
            action_dim=128  # Latent dimension
        )
        
        # Low-level flow (refines individual actions)
        self.low_level_flow = FlowMatching(
            condition_dim=128 + condition_dim,
            action_dim=action_dim
        )
        
        # Sequence decoder
        self.sequence_decoder = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )
        
    def generate(self, condition):
        """Generate action sequence"""
        batch_size = condition.shape[0]
        
        # Generate high-level plan
        high_level_latent = self.high_level_flow.generate(condition)
        
        # Decode to sequence
        latent_sequence = high_level_latent.unsqueeze(1).repeat(1, self.sequence_length, 1)
        sequence_features, _ = self.sequence_decoder(latent_sequence)
        
        # Generate refined actions for each timestep
        actions = []
        for t in range(self.sequence_length):
            # Combine high-level and condition
            step_condition = torch.cat([
                sequence_features[:, t],
                condition
            ], dim=-1)
            
            # Generate action
            action = self.low_level_flow.generate(step_condition)
            actions.append(action)
        
        actions = torch.stack(actions, dim=1)
        
        return actions
```

---

## 🔬 핵심 개념 정리

### 1. Flow Properties
```python
def analyze_flow_properties(flow_model, test_data):
    """Analyze properties of learned flow"""
    results = {}
    
    # Invertibility check
    x = test_data
    z, log_det_forward = flow_model.forward(x)
    x_reconstructed, log_det_inverse = flow_model.inverse(z)
    
    reconstruction_error = (x - x_reconstructed).abs().mean()
    results['reconstruction_error'] = reconstruction_error.item()
    
    # Determinant consistency
    det_consistency = (log_det_forward + log_det_inverse).abs().mean()
    results['determinant_consistency'] = det_consistency.item()
    
    # Likelihood computation
    log_likelihood = flow_model.log_likelihood(x).mean()
    results['log_likelihood'] = log_likelihood.item()
    
    # Sample quality (using FID or similar)
    samples = flow_model.sample(1000)
    results['sample_mean'] = samples.mean().item()
    results['sample_std'] = samples.std().item()
    
    return results
```

### 2. Training Stability
```python
class FlowTrainer:
    """Stable training for flow models"""
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Learning rate scheduling
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )
        
        # Gradient clipping threshold
        self.grad_clip = 1.0
        
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        
        # Forward pass
        if isinstance(self.model, FlowMatching):
            # Flow matching loss
            loss = self.model.compute_loss(
                batch['x_0'], batch['x_1'], batch['condition']
            )
        else:
            # Maximum likelihood loss
            log_likelihood = self.model.log_likelihood(batch)
            loss = -log_likelihood.mean()
        
        # Add regularization
        reg_loss = self.compute_regularization()
        total_loss = loss + 0.01 * reg_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def compute_regularization(self):
        """Regularization for stable training"""
        reg_loss = 0
        
        # Weight decay
        for param in self.model.parameters():
            reg_loss += 0.01 * param.pow(2).sum()
        
        # Spectral normalization (optional)
        # reg_loss += spectral_regularization(self.model)
        
        return reg_loss
```

---

## 🛠️ 실습 코드

### 완전한 Flow-based VLA 시스템
```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class CompleteFlowVLA:
    """완전한 Flow 기반 VLA 시스템"""
    def __init__(self, device='cuda'):
        self.device = device
        
        # Initialize encoders
        self.vision_encoder = self._build_vision_encoder().to(device)
        self.language_encoder = self._build_language_encoder().to(device)
        
        # Initialize flow model
        self.flow_vla = FlowVLA(
            self.vision_encoder,
            self.language_encoder,
            action_dim=7
        ).to(device)
        
        # Training components
        self.optimizer = torch.optim.AdamW(self.flow_vla.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=1e-3,
            total_steps=10000
        )
        
        # Metrics tracking
        self.metrics = {
            'loss': [],
            'action_error': [],
            'generation_time': []
        }
        
    def _build_vision_encoder(self):
        """Build vision encoder"""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
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
            Lambda(lambda x: x[0][:, -1, :])
        )
    
    def train(self, dataloader, epochs=10):
        """Train flow VLA"""
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_error = 0
            
            for batch in dataloader:
                # Move to device
                images = batch['images'].to(self.device)
                instructions = batch['instructions'].to(self.device)
                expert_actions = batch['actions'].to(self.device)
                
                # Training step
                loss = self.flow_vla.train_step({
                    'images': images,
                    'instructions': instructions,
                    'actions': expert_actions
                })
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow_vla.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                epoch_loss += loss.item()
                
                # Evaluate generation
                with torch.no_grad():
                    generated_actions = self.flow_vla(images, instructions)
                    error = (generated_actions - expert_actions).abs().mean()
                    epoch_error += error.item()
            
            # Log metrics
            avg_loss = epoch_loss / len(dataloader)
            avg_error = epoch_error / len(dataloader)
            
            self.metrics['loss'].append(avg_loss)
            self.metrics['action_error'].append(avg_error)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Action Error: {avg_error:.4f}")
    
    def generate_action(self, image, instruction, n_steps=50):
        """Generate action using flow"""
        import time
        
        self.flow_vla.eval()
        
        start_time = time.time()
        
        with torch.no_grad():
            # Move inputs to device
            image = image.to(self.device)
            instruction = instruction.to(self.device)
            
            # Generate action
            action = self.flow_vla(image, instruction)
        
        generation_time = time.time() - start_time
        self.metrics['generation_time'].append(generation_time)
        
        return action.cpu(), generation_time
    
    def visualize_flow(self, condition, n_samples=100):
        """Visualize flow transformation"""
        self.flow_vla.eval()
        
        with torch.no_grad():
            # Sample initial noise
            z = torch.randn(n_samples, 7).to(self.device)
            
            # Track trajectory
            trajectory = [z.cpu().numpy()]
            
            # Generate through flow
            dt = 1.0 / 20
            for i in range(20):
                t = torch.ones(n_samples, 1).to(self.device) * (i * dt)
                v = self.flow_vla.action_flow.forward(z, t, condition)
                z = z + v * dt
                trajectory.append(z.cpu().numpy())
        
        trajectory = np.array(trajectory)
        
        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot first 3 dimensions over time
        for dim in range(3):
            axes[0, dim].plot(trajectory[:, :, dim].T, alpha=0.3)
            axes[0, dim].set_title(f'Dimension {dim+1} Evolution')
            axes[0, dim].set_xlabel('Time Step')
            axes[0, dim].set_ylabel('Value')
        
        # Plot pairwise projections at final time
        for i, (d1, d2) in enumerate([(0, 1), (1, 2), (0, 2)]):
            axes[1, i].scatter(trajectory[-1, :, d1], trajectory[-1, :, d2], alpha=0.5)
            axes[1, i].set_xlabel(f'Dim {d1+1}')
            axes[1, i].set_ylabel(f'Dim {d2+1}')
            axes[1, i].set_title(f'Final Distribution (Dims {d1+1}-{d2+1})')
        
        plt.suptitle('Flow Transformation Visualization')
        plt.tight_layout()
        plt.show()
    
    def compare_with_baselines(self, test_data):
        """Compare flow with baseline methods"""
        results = {}
        
        # Flow-based generation
        flow_actions = []
        flow_times = []
        
        for batch in test_data:
            action, time = self.generate_action(
                batch['image'], batch['instruction']
            )
            flow_actions.append(action)
            flow_times.append(time)
        
        results['flow'] = {
            'mean_time': np.mean(flow_times),
            'std_time': np.std(flow_times)
        }
        
        # Could compare with:
        # - Diffusion policy
        # - Direct regression
        # - VAE-based generation
        
        return results
    
    def save_model(self, path):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.flow_vla.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics
        }, path)
    
    def load_model(self, path):
        """Load trained model"""
        checkpoint = torch.load(path)
        self.flow_vla.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics = checkpoint['metrics']

class Lambda(nn.Module):
    """Lambda layer for functional operations"""
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def forward(self, x):
        return self.func(x)

# Demo dataset
class FlowVLADataset:
    """Demo dataset for flow VLA"""
    def __init__(self, size=1000):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'images': torch.randn(3, 224, 224),
            'instructions': torch.randint(0, 10000, (20,)),
            'actions': torch.randn(7)
        }

# Visualization utilities
def plot_training_curves(metrics):
    """Plot training progress"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curve
    axes[0].plot(metrics['loss'])
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    # Action error
    axes[1].plot(metrics['action_error'])
    axes[1].set_title('Action Generation Error')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('L1 Error')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def visualize_action_distribution(generated_actions, expert_actions):
    """Compare generated vs expert action distributions"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(min(7, generated_actions.shape[-1])):
        row = i // 4
        col = i % 4
        
        axes[row, col].hist(expert_actions[:, i], alpha=0.5, label='Expert', bins=30)
        axes[row, col].hist(generated_actions[:, i], alpha=0.5, label='Generated', bins=30)
        axes[row, col].set_title(f'Action Dimension {i+1}')
        axes[row, col].legend()
    
    # Hide extra subplot
    axes[1, 3].axis('off')
    
    plt.suptitle('Action Distribution Comparison')
    plt.tight_layout()
    plt.show()

# Main demo
if __name__ == "__main__":
    # Initialize system
    system = CompleteFlowVLA(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    from torch.utils.data import DataLoader
    dataset = FlowVLADataset(size=1000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train
    print("Training Flow VLA...")
    system.train(dataloader, epochs=10)
    
    # Plot training curves
    plot_training_curves(system.metrics)
    
    # Test generation
    test_image = torch.randn(1, 3, 224, 224)
    test_instruction = torch.randint(0, 10000, (1, 20))
    
    action, gen_time = system.generate_action(test_image, test_instruction)
    print(f"Generated action: {action.squeeze()}")
    print(f"Generation time: {gen_time:.3f}s")
    
    # Visualize flow
    test_condition = torch.randn(1, 256).to(system.device)
    system.visualize_flow(test_condition)
    
    # Save model
    system.save_model("flow_vla_model.pth")
    print("Model saved!")
```

---

## 📈 다음 단계

### 1. 고급 Flow 기법
- **Glow**: Invertible 1x1 convolutions
- **FFJORD**: Continuous normalizing flows
- **Coupling Flows**: 더 복잡한 coupling layers

### 2. VLA 특화 개선
- **Conditional Flows**: 조건부 생성
- **Hierarchical Flows**: 계층적 행동 생성
- **Multi-modal Flows**: 다중 모달리티 처리

### 3. 최신 연구
- **Score Matching**: Score 기반 생성
- **Consistency Models**: 빠른 생성
- **Rectified Flows**: 개선된 flow matching

---

## 💡 핵심 포인트

### ✅ 기억해야 할 것들
1. **Invertibility**: 정방향/역방향 변환 가능
2. **Exact likelihood**: 정확한 확률 계산
3. **Continuous generation**: 부드러운 생성
4. **Stable training**: 안정적인 학습

### ⚠️ 주의사항
1. **Computational cost**: 계산 비용 높음
2. **Architecture constraints**: 가역성 제약
3. **Memory requirements**: 큰 메모리 필요

### 🎯 VLA 적용 시
1. **Smooth actions**: 부드러운 로봇 동작
2. **Fast generation**: 실시간 생성 가능
3. **Controllability**: 세밀한 제어 가능

---

**🎉 축하합니다! AI Fundamentals 모든 문서 완성!**