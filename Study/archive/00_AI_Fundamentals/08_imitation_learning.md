# 🎭 Imitation Learning: 시연을 통한 학습

**목표**: Behavioral Cloning, Inverse RL, 그리고 로봇 시연 데이터 활용법 이해  
**시간**: 2-3시간  
**전제조건**: 01_neural_networks_basics.md  

---

## 🎯 개발자를 위한 직관적 이해

### Imitation Learning이란?
```python
learning_paradigms = {
    "reinforcement_learning": "시행착오를 통한 학습 (느림)",
    "imitation_learning": "전문가 시연을 보고 학습 (빠름)",
    "supervised_learning": "정답 라벨이 있는 학습"
}

# Imitation Learning의 핵심
imitation_core = {
    "input": "전문가의 (상태, 행동) 쌍",
    "goal": "전문가처럼 행동하는 정책 학습",
    "advantage": "보상 함수 설계 불필요"
}

# 로봇에서의 활용
robot_imitation = {
    "teleoperation": "인간이 로봇을 조작한 데이터",
    "kinesthetic_teaching": "로봇 팔을 직접 움직여 가르침",
    "visual_demonstrations": "비디오를 보고 학습"
}
```

---

## 🏗️ 기본 구조 및 구현

### 1. Behavioral Cloning (BC)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

class BehavioralCloning(nn.Module):
    """가장 간단한 모방 학습: 지도 학습으로 행동 복제"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Optional: Variance prediction for stochastic policy
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state, deterministic=False):
        mean = self.policy(state)
        
        if deterministic:
            return mean
        else:
            # Stochastic policy with learned variance
            std = self.log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            return action, log_prob, mean
    
    def get_action(self, state, deterministic=False):
        """Get action for deployment"""
        with torch.no_grad():
            if deterministic:
                return self.forward(state, deterministic=True)
            else:
                action, _, _ = self.forward(state, deterministic=False)
                return action

class ExpertDataset(Dataset):
    """전문가 시연 데이터셋"""
    def __init__(self, states, actions, augment=False):
        self.states = torch.FloatTensor(states)
        self.actions = torch.FloatTensor(actions)
        self.augment = augment
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        
        # Optional data augmentation
        if self.augment:
            # Add small noise to states
            state = state + torch.randn_like(state) * 0.01
            # Add small noise to actions (careful not to break constraints)
            action = action + torch.randn_like(action) * 0.001
        
        return state, action

def train_bc(model, dataset, epochs=100, lr=1e-3):
    """Behavioral Cloning 학습"""
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for states, expert_actions in dataloader:
            # Forward pass
            if hasattr(model, 'log_std'):
                predicted_actions, log_probs, means = model(states)
                # MSE loss on means + entropy regularization
                loss = F.mse_loss(means, expert_actions)
                loss = loss - 0.01 * log_probs.mean()  # Entropy bonus
            else:
                predicted_actions = model(states, deterministic=True)
                loss = F.mse_loss(predicted_actions, expert_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model
```

### 2. DAgger (Dataset Aggregation)
```python
class DAgger:
    """Interactive imitation learning to handle distribution shift"""
    def __init__(self, expert, environment, state_dim, action_dim):
        self.expert = expert
        self.env = environment
        self.policy = BehavioralCloning(state_dim, action_dim)
        
        # Aggregated dataset
        self.states = []
        self.actions = []
        
    def collect_expert_data(self, num_episodes=10):
        """Collect initial expert demonstrations"""
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                # Expert action
                action = self.expert.get_action(state)
                
                # Store data
                self.states.append(state)
                self.actions.append(action)
                
                # Step environment
                state, reward, done, _ = self.env.step(action)
        
        return np.array(self.states), np.array(self.actions)
    
    def train_iteration(self, beta=0.5):
        """One iteration of DAgger"""
        # Train policy on current dataset
        dataset = ExpertDataset(self.states, self.actions)
        self.policy = train_bc(self.policy, dataset, epochs=50)
        
        # Collect new data with mixed policy
        new_states = []
        new_actions = []
        
        for _ in range(5):  # Collect 5 episodes
            state = self.env.reset()
            done = False
            
            while not done:
                # Mix expert and learned policy
                if np.random.random() < beta:
                    # Use expert
                    action = self.expert.get_action(state)
                else:
                    # Use learned policy
                    action = self.policy.get_action(
                        torch.FloatTensor(state).unsqueeze(0)
                    ).squeeze(0).numpy()
                
                # Get expert label for this state
                expert_action = self.expert.get_action(state)
                
                # Store state with expert action
                new_states.append(state)
                new_actions.append(expert_action)
                
                # Step with actual action
                state, reward, done, _ = self.env.step(action)
        
        # Aggregate data
        self.states.extend(new_states)
        self.actions.extend(new_actions)
        
        return self.policy
    
    def train(self, iterations=10):
        """Full DAgger training"""
        # Initial expert data collection
        self.collect_expert_data()
        
        # Iterative training
        for i in range(iterations):
            beta = 0.5 ** i  # Decay expert usage
            self.policy = self.train_iteration(beta)
            
            # Evaluate
            success_rate = self.evaluate()
            print(f"Iteration {i+1}, Success rate: {success_rate:.2%}")
        
        return self.policy
    
    def evaluate(self, num_episodes=10):
        """Evaluate learned policy"""
        successes = 0
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.policy.get_action(
                    torch.FloatTensor(state).unsqueeze(0)
                ).squeeze(0).numpy()
                state, reward, done, info = self.env.step(action)
                
            if info.get('success', False):
                successes += 1
        
        return successes / num_episodes
```

### 3. Inverse Reinforcement Learning (IRL)
```python
class MaxEntIRL(nn.Module):
    """Maximum Entropy Inverse Reinforcement Learning"""
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        
        # Reward function network
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Value function for soft Bellman backup
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def get_reward(self, state):
        """Compute learned reward"""
        return self.reward_net(state)
    
    def get_value(self, state):
        """Compute soft value function"""
        return self.value_net(state)
    
    def compute_soft_bellman_backup(self, states, next_states, dones, gamma=0.99):
        """Soft Bellman backup for MaxEnt IRL"""
        rewards = self.get_reward(states)
        next_values = self.get_value(next_states)
        
        # Soft Bellman equation
        # V(s) = r(s) + γ * log E[exp(V(s'))]
        targets = rewards + gamma * (1 - dones.float()) * next_values
        
        return targets
    
    def compute_expert_value(self, expert_trajectories):
        """Compute value of expert trajectories"""
        total_value = 0
        for trajectory in expert_trajectories:
            states = torch.FloatTensor(trajectory['states'])
            rewards = self.get_reward(states)
            
            # Discounted sum
            discounted_reward = 0
            gamma = 0.99
            for t, r in enumerate(rewards):
                discounted_reward += (gamma ** t) * r
            
            total_value += discounted_reward
        
        return total_value / len(expert_trajectories)

def train_irl(irl_model, expert_trajectories, policy_trajectories, epochs=100):
    """Train IRL model"""
    optimizer = torch.optim.Adam(irl_model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        # Sample batch of trajectories
        expert_batch = np.random.choice(expert_trajectories, size=32)
        policy_batch = np.random.choice(policy_trajectories, size=32)
        
        # Compute values
        expert_value = irl_model.compute_expert_value(expert_batch)
        policy_value = irl_model.compute_expert_value(policy_batch)
        
        # MaxEnt IRL loss: maximize difference
        loss = -torch.mean(expert_value - policy_value)
        
        # Add gradient penalty for stability
        grad_penalty = compute_gradient_penalty(irl_model, expert_batch)
        loss = loss + 10 * grad_penalty
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return irl_model

def compute_gradient_penalty(model, real_data, lambda_gp=10):
    """Gradient penalty for training stability"""
    real_data = torch.FloatTensor(real_data).requires_grad_(True)
    
    rewards = model.get_reward(real_data)
    
    gradients = torch.autograd.grad(
        outputs=rewards,
        inputs=real_data,
        grad_outputs=torch.ones_like(rewards),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty
```

### 4. GAIL (Generative Adversarial Imitation Learning)
```python
class GAIL(nn.Module):
    """Generative Adversarial Imitation Learning"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Discriminator (classifies expert vs policy)
        self.discriminator = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Generator (policy)
        self.policy = BehavioralCloning(state_dim, action_dim, hidden_dim)
        
    def discriminate(self, states, actions):
        """Discriminator forward pass"""
        sa_pairs = torch.cat([states, actions], dim=-1)
        return self.discriminator(sa_pairs)
    
    def compute_reward(self, states, actions):
        """Use discriminator output as reward"""
        with torch.no_grad():
            d_values = self.discriminate(states, actions)
            # GAIL reward: -log(1 - D(s,a))
            rewards = -torch.log(1 - d_values + 1e-8)
        return rewards

class GAILTrainer:
    """GAIL training loop"""
    def __init__(self, env, expert_data, state_dim, action_dim):
        self.env = env
        self.expert_states = torch.FloatTensor(expert_data['states'])
        self.expert_actions = torch.FloatTensor(expert_data['actions'])
        
        self.gail = GAIL(state_dim, action_dim)
        self.d_optimizer = torch.optim.Adam(self.gail.discriminator.parameters(), lr=1e-3)
        self.g_optimizer = torch.optim.Adam(self.gail.policy.parameters(), lr=1e-3)
        
    def train_discriminator(self, policy_data, n_epochs=5):
        """Train discriminator to distinguish expert from policy"""
        for _ in range(n_epochs):
            # Sample batch
            batch_size = min(256, len(self.expert_states))
            expert_idx = np.random.choice(len(self.expert_states), batch_size)
            expert_batch_s = self.expert_states[expert_idx]
            expert_batch_a = self.expert_actions[expert_idx]
            
            policy_idx = np.random.choice(len(policy_data['states']), batch_size)
            policy_batch_s = torch.FloatTensor(policy_data['states'][policy_idx])
            policy_batch_a = torch.FloatTensor(policy_data['actions'][policy_idx])
            
            # Discriminator predictions
            expert_preds = self.gail.discriminate(expert_batch_s, expert_batch_a)
            policy_preds = self.gail.discriminate(policy_batch_s, policy_batch_a)
            
            # Binary cross entropy loss
            expert_loss = F.binary_cross_entropy(expert_preds, torch.ones_like(expert_preds))
            policy_loss = F.binary_cross_entropy(policy_preds, torch.zeros_like(policy_preds))
            d_loss = expert_loss + policy_loss
            
            # Update discriminator
            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()
        
        return d_loss.item()
    
    def train_generator(self, n_steps=1000):
        """Train generator (policy) with PPO"""
        states = []
        actions = []
        rewards = []
        
        # Collect trajectories
        state = self.env.reset()
        for _ in range(n_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, _ = self.gail.policy(state_tensor)
            action = action.squeeze(0).detach().numpy()
            
            # Get reward from discriminator
            reward = self.gail.compute_reward(
                state_tensor, 
                torch.FloatTensor(action).unsqueeze(0)
            ).item()
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state, _, done, _ = self.env.step(action)
            if done:
                state = self.env.reset()
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        
        # Compute returns (discounted cumulative rewards)
        returns = self.compute_returns(rewards)
        
        # Policy gradient update
        action_preds, log_probs, _ = self.gail.policy(states)
        
        # Advantage = returns - baseline (simplified)
        advantages = returns - returns.mean()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy gradient loss
        g_loss = -(log_probs * advantages).mean()
        
        # Update generator
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()
        
        return {'states': states.numpy(), 'actions': actions.numpy()}, g_loss.item()
    
    def compute_returns(self, rewards, gamma=0.99):
        """Compute discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def train(self, iterations=100):
        """Full GAIL training"""
        for iteration in range(iterations):
            # Generate trajectories with current policy
            policy_data, g_loss = self.train_generator()
            
            # Train discriminator
            d_loss = self.train_discriminator(policy_data)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}")
        
        return self.gail.policy
```

---

## 🤖 VLA에서의 활용

### 1. Visual Imitation for VLA
```python
class VisualImitationVLA(nn.Module):
    """비전 기반 모방 학습 VLA"""
    def __init__(self, vision_encoder, action_dim=7):
        super().__init__()
        self.vision_encoder = vision_encoder
        
        # Temporal encoder for video demonstrations
        self.temporal_encoder = nn.LSTM(
            input_size=768,
            hidden_size=512,
            num_layers=2,
            batch_first=True
        )
        
        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Keyframe attention
        self.keyframe_attention = nn.MultiheadAttention(512, 8, batch_first=True)
        
    def forward(self, video_frames):
        """Process video demonstration"""
        batch_size, seq_len = video_frames.shape[:2]
        
        # Encode each frame
        frame_features = []
        for t in range(seq_len):
            features = self.vision_encoder(video_frames[:, t])
            frame_features.append(features)
        
        frame_features = torch.stack(frame_features, dim=1)
        
        # Temporal encoding
        temporal_features, (hidden, cell) = self.temporal_encoder(frame_features)
        
        # Keyframe attention (attend to important frames)
        attended_features, attention_weights = self.keyframe_attention(
            temporal_features, temporal_features, temporal_features
        )
        
        # Decode to actions
        actions = self.action_decoder(attended_features)
        
        return actions, attention_weights
    
    def extract_keyframes(self, video, threshold=0.8):
        """Extract keyframes from demonstration"""
        with torch.no_grad():
            _, attention_weights = self.forward(video.unsqueeze(0))
            
        # Find frames with high attention
        frame_importance = attention_weights.mean(dim=(0, 1))
        keyframe_indices = torch.where(frame_importance > threshold)[0]
        
        return keyframe_indices

class OneShotImitationVLA(nn.Module):
    """One-shot imitation learning VLA"""
    def __init__(self, vision_encoder, language_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_encoder = language_encoder
        
        # Meta-learning components
        self.task_encoder = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Adaptation network
        self.adaptation_net = nn.Sequential(
            nn.Linear(256 + 768, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(256 + 768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )
        
    def encode_demonstration(self, demo_video, demo_instruction):
        """Encode single demonstration"""
        # Average pool video features
        video_features = []
        for frame in demo_video:
            features = self.vision_encoder(frame.unsqueeze(0))
            video_features.append(features)
        
        video_encoding = torch.stack(video_features).mean(dim=0)
        
        # Encode instruction
        instruction_encoding = self.language_encoder(demo_instruction)
        
        # Combine into task representation
        combined = torch.cat([video_encoding, instruction_encoding], dim=-1)
        task_representation = self.task_encoder(combined)
        
        return task_representation
    
    def forward(self, current_obs, instruction, demo_representation):
        """Execute task given demonstration"""
        # Encode current state
        obs_features = self.vision_encoder(current_obs)
        inst_features = self.language_encoder(instruction)
        
        # Adapt to demonstration
        adaptation_input = torch.cat([demo_representation, obs_features], dim=-1)
        adapted_features = self.adaptation_net(adaptation_input)
        
        # Generate action
        policy_input = torch.cat([adapted_features, inst_features], dim=-1)
        action = self.policy(policy_input)
        
        return action
```

### 2. Trajectory Optimization from Demonstrations
```python
class TrajectoryOptimizationIL:
    """시연으로부터 궤적 최적화"""
    def __init__(self, dynamics_model, cost_function):
        self.dynamics = dynamics_model
        self.cost = cost_function
        
    def optimize_from_demo(self, demo_trajectory, horizon=50):
        """Optimize trajectory starting from demonstration"""
        # Initialize with demonstration
        trajectory = demo_trajectory.clone()
        
        # Iterative optimization
        for iteration in range(10):
            # Compute gradients
            gradients = self.compute_trajectory_gradient(trajectory)
            
            # Line search
            alpha = self.line_search(trajectory, gradients)
            
            # Update trajectory
            trajectory = trajectory - alpha * gradients
            
            # Project to feasible space
            trajectory = self.project_to_constraints(trajectory)
        
        return trajectory
    
    def compute_trajectory_gradient(self, trajectory):
        """Compute gradient of cost w.r.t. trajectory"""
        trajectory.requires_grad_(True)
        
        total_cost = 0
        states = [trajectory[0, :7]]  # Initial state
        
        for t in range(len(trajectory) - 1):
            action = trajectory[t, 7:]
            next_state = self.dynamics(states[-1], action)
            states.append(next_state)
            
            # Accumulate cost
            total_cost += self.cost(states[-1], action)
        
        # Compute gradients
        gradients = torch.autograd.grad(total_cost, trajectory)[0]
        
        return gradients
    
    def line_search(self, trajectory, gradients, alpha_init=0.1):
        """Line search for step size"""
        alpha = alpha_init
        current_cost = self.evaluate_trajectory(trajectory)
        
        while alpha > 1e-6:
            new_trajectory = trajectory - alpha * gradients
            new_cost = self.evaluate_trajectory(new_trajectory)
            
            if new_cost < current_cost:
                return alpha
            
            alpha *= 0.5
        
        return alpha
    
    def evaluate_trajectory(self, trajectory):
        """Evaluate total cost of trajectory"""
        total_cost = 0
        state = trajectory[0, :7]
        
        for t in range(len(trajectory) - 1):
            action = trajectory[t, 7:]
            state = self.dynamics(state, action)
            total_cost += self.cost(state, action).item()
        
        return total_cost
    
    def project_to_constraints(self, trajectory):
        """Project trajectory to satisfy constraints"""
        # Joint limits
        trajectory[:, :7] = torch.clamp(trajectory[:, :7], -np.pi, np.pi)
        
        # Velocity limits
        trajectory[:, 7:] = torch.clamp(trajectory[:, 7:], -1.0, 1.0)
        
        return trajectory
```

---

## 🔬 핵심 개념 정리

### 1. Distribution Shift 문제와 해결
```python
def analyze_distribution_shift(expert_data, policy_rollouts):
    """분포 변화 분석"""
    # Compute state distributions
    expert_states = expert_data['states']
    policy_states = policy_rollouts['states']
    
    # KL divergence
    expert_mean = expert_states.mean(axis=0)
    expert_std = expert_states.std(axis=0)
    
    policy_mean = policy_states.mean(axis=0)
    policy_std = policy_states.std(axis=0)
    
    kl_div = 0.5 * np.sum(
        np.log(policy_std / expert_std) + 
        (expert_std**2 + (expert_mean - policy_mean)**2) / (2 * policy_std**2) - 0.5
    )
    
    return {
        'kl_divergence': kl_div,
        'expert_coverage': compute_state_coverage(expert_states),
        'policy_coverage': compute_state_coverage(policy_states)
    }

def compute_state_coverage(states, grid_size=10):
    """Compute state space coverage"""
    # Discretize state space
    min_vals = states.min(axis=0)
    max_vals = states.max(axis=0)
    
    bins = []
    for i in range(states.shape[1]):
        bins.append(np.linspace(min_vals[i], max_vals[i], grid_size))
    
    # Count occupied cells
    hist, _ = np.histogramdd(states, bins=bins)
    coverage = (hist > 0).sum() / (grid_size ** states.shape[1])
    
    return coverage
```

### 2. Reward Recovery Evaluation
```python
def evaluate_reward_recovery(learned_reward, true_reward, test_states):
    """평가: 학습된 보상 함수의 품질"""
    learned_values = learned_reward(test_states)
    true_values = true_reward(test_states)
    
    # Correlation
    correlation = np.corrcoef(learned_values.flatten(), true_values.flatten())[0, 1]
    
    # MSE
    mse = np.mean((learned_values - true_values) ** 2)
    
    # Ranking accuracy
    learned_ranks = np.argsort(learned_values.flatten())
    true_ranks = np.argsort(true_values.flatten())
    rank_correlation = np.corrcoef(learned_ranks, true_ranks)[0, 1]
    
    return {
        'correlation': correlation,
        'mse': mse,
        'rank_correlation': rank_correlation
    }
```

---

## 🛠️ 실습 코드

### 완전한 로봇 모방 학습 시스템
```python
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import gym

class RobotImitationSystem:
    """완전한 로봇 모방 학습 시스템"""
    def __init__(self, env_name='RobotArm-v0'):
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # Models
        self.bc_policy = BehavioralCloning(self.state_dim, self.action_dim)
        self.gail_model = GAIL(self.state_dim, self.action_dim)
        
        # Data storage
        self.expert_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        
        self.success_threshold = 0.9
        
    def collect_teleoperation_data(self, num_episodes=50):
        """텔레오퍼레이션으로 전문가 데이터 수집"""
        print("Collecting expert demonstrations...")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            trajectory = {'states': [], 'actions': [], 'rewards': []}
            done = False
            
            while not done:
                # Get expert action (simulated here, replace with actual teleoperation)
                action = self.get_expert_action(state)
                
                # Store transition
                trajectory['states'].append(state)
                trajectory['actions'].append(action)
                
                # Step environment
                next_state, reward, done, _ = self.env.step(action)
                trajectory['rewards'].append(reward)
                
                state = next_state
            
            # Store successful trajectories
            if sum(trajectory['rewards']) > self.success_threshold:
                self.expert_buffer['states'].extend(trajectory['states'])
                self.expert_buffer['actions'].extend(trajectory['actions'])
                self.expert_buffer['rewards'].extend(trajectory['rewards'])
        
        # Convert to numpy arrays
        for key in self.expert_buffer:
            self.expert_buffer[key] = np.array(self.expert_buffer[key])
        
        print(f"Collected {len(self.expert_buffer['states'])} expert transitions")
        
    def get_expert_action(self, state):
        """Simulated expert policy (replace with actual)"""
        # Simple PD controller to target
        target = np.array([0.5, 0.5, 0.5])  # Target position
        error = target - state[:3]
        action = 2.0 * error - 0.5 * state[3:6]  # P and D terms
        return np.clip(action, -1, 1)
    
    def train_bc(self, epochs=100):
        """Train with behavioral cloning"""
        print("Training with Behavioral Cloning...")
        
        dataset = ExpertDataset(
            self.expert_buffer['states'],
            self.expert_buffer['actions'],
            augment=True
        )
        
        self.bc_policy = train_bc(self.bc_policy, dataset, epochs=epochs)
        
        # Evaluate
        success_rate = self.evaluate_policy(self.bc_policy)
        print(f"BC Success Rate: {success_rate:.2%}")
        
        return self.bc_policy
    
    def train_dagger(self, iterations=10):
        """Train with DAgger"""
        print("Training with DAgger...")
        
        dagger = DAgger(
            expert=self,  # Use self as expert
            environment=self.env,
            state_dim=self.state_dim,
            action_dim=self.action_dim
        )
        
        # Use collected expert data
        dagger.states = list(self.expert_buffer['states'])
        dagger.actions = list(self.expert_buffer['actions'])
        
        policy = dagger.train(iterations=iterations)
        
        return policy
    
    def train_gail(self, iterations=100):
        """Train with GAIL"""
        print("Training with GAIL...")
        
        trainer = GAILTrainer(
            env=self.env,
            expert_data=self.expert_buffer,
            state_dim=self.state_dim,
            action_dim=self.action_dim
        )
        
        policy = trainer.train(iterations=iterations)
        
        # Evaluate
        success_rate = self.evaluate_policy(policy)
        print(f"GAIL Success Rate: {success_rate:.2%}")
        
        return policy
    
    def evaluate_policy(self, policy, num_episodes=20):
        """Evaluate learned policy"""
        successes = 0
        total_rewards = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Get action from policy
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                if hasattr(policy, 'get_action'):
                    action = policy.get_action(state_tensor, deterministic=True)
                else:
                    action = policy(state_tensor, deterministic=True)
                
                action = action.squeeze(0).detach().numpy()
                
                # Step environment
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            if episode_reward > self.success_threshold:
                successes += 1
        
        return successes / num_episodes
    
    def compare_methods(self):
        """Compare different IL methods"""
        results = {}
        
        # Behavioral Cloning
        bc_policy = self.train_bc()
        results['BC'] = self.evaluate_policy(bc_policy)
        
        # DAgger
        dagger_policy = self.train_dagger()
        results['DAgger'] = self.evaluate_policy(dagger_policy)
        
        # GAIL
        gail_policy = self.train_gail()
        results['GAIL'] = self.evaluate_policy(gail_policy)
        
        # Print comparison
        print("\n=== Method Comparison ===")
        for method, success_rate in results.items():
            print(f"{method}: {success_rate:.2%}")
        
        return results
    
    def save_policy(self, policy, path):
        """Save trained policy"""
        torch.save({
            'model_state_dict': policy.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, path)
    
    def load_policy(self, path):
        """Load trained policy"""
        checkpoint = torch.load(path)
        policy = BehavioralCloning(
            checkpoint['state_dim'],
            checkpoint['action_dim']
        )
        policy.load_state_dict(checkpoint['model_state_dict'])
        return policy

# Visualization utilities
def visualize_demonstrations(expert_data, learned_rollouts):
    """Visualize expert vs learned trajectories"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # State distributions
    for i in range(2):
        axes[0, i].hist(expert_data['states'][:, i], alpha=0.5, label='Expert', bins=30)
        axes[0, i].hist(learned_rollouts['states'][:, i], alpha=0.5, label='Learned', bins=30)
        axes[0, i].set_xlabel(f'State Dimension {i}')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].legend()
    
    # Action distributions
    for i in range(2):
        axes[1, i].hist(expert_data['actions'][:, i], alpha=0.5, label='Expert', bins=30)
        axes[1, i].hist(learned_rollouts['actions'][:, i], alpha=0.5, label='Learned', bins=30)
        axes[1, i].set_xlabel(f'Action Dimension {i}')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].legend()
    
    plt.suptitle('Expert vs Learned Distributions')
    plt.tight_layout()
    plt.show()

def plot_learning_curves(bc_losses, gail_d_losses, gail_g_losses):
    """Plot training curves"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # BC loss
    axes[0].plot(bc_losses)
    axes[0].set_title('BC Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    
    # GAIL discriminator loss
    axes[1].plot(gail_d_losses)
    axes[1].set_title('GAIL Discriminator Loss')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Binary Cross Entropy')
    
    # GAIL generator loss
    axes[2].plot(gail_g_losses)
    axes[2].set_title('GAIL Generator Loss')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Policy Gradient Loss')
    
    plt.tight_layout()
    plt.show()

# Demo usage
if __name__ == "__main__":
    # Initialize system
    system = RobotImitationSystem()
    
    # Collect expert data
    system.collect_teleoperation_data(num_episodes=50)
    
    # Compare methods
    results = system.compare_methods()
    
    # Save best policy
    best_method = max(results, key=results.get)
    print(f"\nBest method: {best_method} with {results[best_method]:.2%} success rate")
```

---

## 📈 다음 단계

### 1. 고급 모방 학습 기법
- **Adversarial Inverse RL**: 더 강건한 보상 학습
- **Meta Imitation Learning**: Few-shot 모방
- **Hierarchical Imitation**: 계층적 스킬 학습

### 2. VLA 특화 개선
- **Language-Conditioned IL**: 언어 지시 모방
- **Multi-Modal Imitation**: 비전+언어+촉각
- **Interactive Imitation**: 실시간 교정

### 3. 실용적 고려사항
- **Safety in Imitation**: 안전한 모방
- **Sim-to-Real Transfer**: 시뮬레이션→실제
- **Data Efficiency**: 적은 시연으로 학습

---

## 💡 핵심 포인트

### ✅ 기억해야 할 것들
1. **BC는 간단하지만 distribution shift 문제**
2. **DAgger는 interactive하게 해결**
3. **IRL은 보상 함수를 학습**
4. **GAIL은 adversarial 방식 활용**

### ⚠️ 주의사항
1. **Covariate shift**: 훈련/테스트 분포 차이
2. **Compounding errors**: 에러 누적
3. **Expert suboptimality**: 완벽하지 않은 시연

### 🎯 VLA 적용 시
1. **Multi-modal demonstrations**: 다양한 센서 활용
2. **Keyframe extraction**: 중요 순간 추출
3. **Safety constraints**: 안전 제약 준수

---

**다음 문서**: `12_cross_modal_alignment.md` - 모달리티 간 정렬