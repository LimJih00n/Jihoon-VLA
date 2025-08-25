# 🎮 Reinforcement Learning: 시행착오를 통한 학습

**목표**: MDP, Policy Gradient, Actor-Critic, 그리고 로봇 제어를 위한 RL 이해  
**시간**: 3-4시간  
**전제조건**: 01_neural_networks_basics.md  

---

## 🎯 개발자를 위한 직관적 이해

### Reinforcement Learning의 핵심
```python
rl_components = {
    "agent": "학습하는 주체 (로봇)",
    "environment": "상호작용하는 세계",
    "state": "현재 상황",
    "action": "할 수 있는 행동",
    "reward": "행동의 결과 (좋음/나쁨)",
    "policy": "상태 → 행동 매핑"
}

# RL의 목표
rl_goal = "누적 보상을 최대화하는 정책 학습"

# 로봇 제어에서의 예시
robot_rl = {
    "state": "카메라 이미지 + 관절 각도",
    "action": "모터 토크 명령",
    "reward": "작업 성공 +1, 충돌 -1",
    "policy": "신경망이 상태를 보고 행동 결정"
}
```

---

## 🏗️ 기본 구조 및 구현

### 1. Markov Decision Process (MDP)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import gym

class MDP:
    """Markov Decision Process 기본 구조"""
    def __init__(self, states, actions, transition_prob, reward_func, gamma=0.99):
        self.states = states
        self.actions = actions
        self.P = transition_prob  # P(s'|s,a)
        self.R = reward_func      # R(s,a,s')
        self.gamma = gamma        # Discount factor
        
    def value_iteration(self, threshold=1e-6):
        """Value iteration algorithm"""
        V = np.zeros(len(self.states))
        
        while True:
            V_old = V.copy()
            
            for s in range(len(self.states)):
                # Bellman optimality equation
                q_values = []
                for a in range(len(self.actions)):
                    q_value = 0
                    for s_next in range(len(self.states)):
                        # Q(s,a) = sum over s' of P(s'|s,a) * [R(s,a,s') + γV(s')]
                        q_value += self.P[s, a, s_next] * (
                            self.R[s, a, s_next] + self.gamma * V_old[s_next]
                        )
                    q_values.append(q_value)
                
                V[s] = max(q_values)
            
            # Check convergence
            if np.max(np.abs(V - V_old)) < threshold:
                break
        
        # Extract policy from value function
        policy = np.zeros(len(self.states), dtype=int)
        for s in range(len(self.states)):
            q_values = []
            for a in range(len(self.actions)):
                q_value = 0
                for s_next in range(len(self.states)):
                    q_value += self.P[s, a, s_next] * (
                        self.R[s, a, s_next] + self.gamma * V[s_next]
                    )
                q_values.append(q_value)
            
            policy[s] = np.argmax(q_values)
        
        return V, policy
    
    def policy_iteration(self):
        """Policy iteration algorithm"""
        # Initialize random policy
        policy = np.random.choice(len(self.actions), size=len(self.states))
        
        while True:
            # Policy evaluation
            V = self.policy_evaluation(policy)
            
            # Policy improvement
            policy_new = np.zeros(len(self.states), dtype=int)
            
            for s in range(len(self.states)):
                q_values = []
                for a in range(len(self.actions)):
                    q_value = 0
                    for s_next in range(len(self.states)):
                        q_value += self.P[s, a, s_next] * (
                            self.R[s, a, s_next] + self.gamma * V[s_next]
                        )
                    q_values.append(q_value)
                
                policy_new[s] = np.argmax(q_values)
            
            # Check if policy has converged
            if np.array_equal(policy, policy_new):
                break
            
            policy = policy_new
        
        return V, policy
    
    def policy_evaluation(self, policy, threshold=1e-6):
        """Evaluate a given policy"""
        V = np.zeros(len(self.states))
        
        while True:
            V_old = V.copy()
            
            for s in range(len(self.states)):
                a = policy[s]
                v = 0
                for s_next in range(len(self.states)):
                    v += self.P[s, a, s_next] * (
                        self.R[s, a, s_next] + self.gamma * V_old[s_next]
                    )
                V[s] = v
            
            if np.max(np.abs(V - V_old)) < threshold:
                break
        
        return V
```

### 2. Deep Q-Network (DQN)
```python
class DQN(nn.Module):
    """Deep Q-Network for continuous states"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state):
        return self.q_network(state)
    
    def get_action(self, state, epsilon=0.1):
        """Epsilon-greedy action selection"""
        if np.random.random() < epsilon:
            # Random action
            return np.random.randint(0, self.q_network[-1].out_features)
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.forward(state)
                return q_values.argmax().item()

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (torch.FloatTensor(state),
                torch.LongTensor(action),
                torch.FloatTensor(reward),
                torch.FloatTensor(next_state),
                torch.FloatTensor(done))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN agent with experience replay and target network"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Training components
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        self.gamma = gamma
        
        # Hyperparameters
        self.batch_size = 32
        self.update_target_every = 1000
        self.steps = 0
        
    def update(self):
        """Update Q-network"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
```

### 3. Policy Gradient (REINFORCE)
```python
class PolicyNetwork(nn.Module):
    """Stochastic policy network"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state):
        logits = self.network(state)
        return F.softmax(logits, dim=-1)
    
    def get_action(self, state):
        """Sample action from policy"""
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob

class REINFORCE:
    """REINFORCE algorithm (Monte Carlo Policy Gradient)"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        
    def compute_returns(self, rewards):
        """Compute discounted returns"""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update(self, trajectories):
        """Update policy using collected trajectories"""
        total_loss = 0
        
        for trajectory in trajectories:
            states = trajectory['states']
            actions = trajectory['actions']
            rewards = trajectory['rewards']
            log_probs = trajectory['log_probs']
            
            # Compute returns
            returns = self.compute_returns(rewards)
            
            # Policy gradient loss
            loss = 0
            for log_prob, G in zip(log_probs, returns):
                loss += -log_prob * G
            
            total_loss += loss
        
        # Average over trajectories
        loss = total_loss / len(trajectories)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def collect_trajectory(self, env, max_steps=1000):
        """Collect single trajectory"""
        state = env.reset()
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': []
        }
        
        for _ in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob = self.policy.get_action(state_tensor)
            
            next_state, reward, done, _ = env.step(action)
            
            trajectory['states'].append(state)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['log_probs'].append(log_prob)
            
            state = next_state
            
            if done:
                break
        
        return trajectory
```

### 4. Actor-Critic
```python
class ActorCritic(nn.Module):
    """Actor-Critic network"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        shared_features = self.shared(state)
        
        # Policy
        action_logits = self.actor(shared_features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Value
        value = self.critic(shared_features)
        
        return action_probs, value
    
    def get_action_and_value(self, state):
        """Get action and value for given state"""
        action_probs, value = self.forward(state)
        
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value

class A2C:
    """Advantage Actor-Critic"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        
    def compute_advantages(self, rewards, values, next_value, dones):
        """Compute advantages using GAE"""
        advantages = []
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
            
            # TD error
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantage = delta + self.gamma * 0.95 * advantage * (1 - dones[t])
            advantages.insert(0, advantage)
            
            next_value = values[t]
        
        return torch.FloatTensor(advantages)
    
    def update(self, trajectories):
        """Update actor and critic"""
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        for traj in trajectories:
            states.extend(traj['states'])
            actions.extend(traj['actions'])
            rewards.extend(traj['rewards'])
            values.extend(traj['values'])
            log_probs.extend(traj['log_probs'])
            dones.extend(traj['dones'])
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        dones = torch.FloatTensor(dones)
        
        # Compute advantages
        with torch.no_grad():
            _, next_value = self.actor_critic(states[-1:])
        
        advantages = self.compute_advantages(rewards, values, next_value, dones)
        returns = advantages + values.squeeze()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute losses
        action_probs, new_values = self.actor_critic(states)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        
        # Actor loss (policy gradient)
        actor_loss = -(new_log_probs * advantages.detach()).mean()
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(new_values.squeeze(), returns.detach())
        
        # Entropy bonus for exploration
        entropy = dist.entropy().mean()
        
        # Total loss
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
        self.optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item()
        }
```

### 5. PPO (Proximal Policy Optimization)
```python
class PPO:
    """PPO - State of the art policy gradient"""
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
    def compute_gae(self, rewards, values, next_value, dones):
        """Generalized Advantage Estimation"""
        advantages = []
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * advantage
            advantages.insert(0, advantage)
        
        advantages = torch.FloatTensor(advantages)
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, advantages, returns, epochs=4, batch_size=64):
        """PPO update with multiple epochs"""
        dataset_size = states.shape[0]
        
        for _ in range(epochs):
            # Shuffle data
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current policy
                action_probs, values = self.actor_critic(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                log_probs = dist.log_prob(batch_actions)
                
                # Compute ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy bonus
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
```

---

## 🤖 VLA에서의 활용

### 1. RL for Robot Manipulation
```python
class RobotManipulationRL:
    """로봇 조작을 위한 RL"""
    def __init__(self, observation_space, action_space):
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        
        # Continuous action space - use SAC
        self.agent = SAC(self.obs_dim, self.act_dim)
        
        # Safety layer
        self.safety_filter = SafetyFilter()
        
        # Reward shaping
        self.reward_shaper = RewardShaper()
        
    def train_episode(self, env):
        """Train single episode with safety"""
        state = env.reset()
        episode_reward = 0
        trajectory = []
        
        while True:
            # Get action from agent
            action = self.agent.get_action(state)
            
            # Apply safety filter
            safe_action = self.safety_filter.filter_action(state, action)
            
            # Step environment
            next_state, reward, done, info = env.step(safe_action)
            
            # Shape reward
            shaped_reward = self.reward_shaper.shape(
                state, safe_action, next_state, reward, info
            )
            
            # Store transition
            self.agent.replay_buffer.push(
                state, safe_action, shaped_reward, next_state, done
            )
            
            # Update agent
            if len(self.agent.replay_buffer) > 1000:
                self.agent.update()
            
            trajectory.append({
                'state': state,
                'action': safe_action,
                'reward': shaped_reward
            })
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        return episode_reward, trajectory

class SafetyFilter:
    """Safety filter for robot actions"""
    def __init__(self, joint_limits, velocity_limits, workspace_bounds):
        self.joint_limits = joint_limits
        self.velocity_limits = velocity_limits
        self.workspace_bounds = workspace_bounds
        
    def filter_action(self, state, action):
        """Filter action for safety"""
        # Clip to joint limits
        action = np.clip(action, self.joint_limits[:, 0], self.joint_limits[:, 1])
        
        # Limit velocities
        current_velocity = state[-7:]  # Assuming last 7 are velocities
        new_velocity = current_velocity + action
        new_velocity = np.clip(new_velocity, -self.velocity_limits, self.velocity_limits)
        action = new_velocity - current_velocity
        
        # Check workspace bounds
        predicted_position = self.forward_kinematics(state[:7] + action)
        if not self.in_workspace(predicted_position):
            # Scale down action
            action *= 0.5
        
        return action
    
    def forward_kinematics(self, joint_angles):
        """Simple forward kinematics (placeholder)"""
        # Would compute actual end-effector position
        return joint_angles[:3]
    
    def in_workspace(self, position):
        """Check if position is in safe workspace"""
        return np.all(position >= self.workspace_bounds[:, 0]) and \
               np.all(position <= self.workspace_bounds[:, 1])

class RewardShaper:
    """Reward shaping for faster learning"""
    def __init__(self):
        self.goal_position = np.array([0.5, 0.5, 0.3])
        self.previous_distance = None
        
    def shape(self, state, action, next_state, reward, info):
        """Shape reward signal"""
        shaped_reward = reward
        
        # Distance-based shaping
        current_pos = next_state[:3]
        distance = np.linalg.norm(current_pos - self.goal_position)
        
        if self.previous_distance is not None:
            # Reward for getting closer
            delta_distance = self.previous_distance - distance
            shaped_reward += 0.1 * delta_distance
        
        self.previous_distance = distance
        
        # Penalty for large actions (energy)
        shaped_reward -= 0.01 * np.linalg.norm(action)
        
        # Bonus for success
        if info.get('success', False):
            shaped_reward += 10.0
        
        return shaped_reward
```

### 2. Hierarchical RL for VLA
```python
class HierarchicalRL:
    """Hierarchical RL for complex tasks"""
    def __init__(self, state_dim, num_skills=10):
        # High-level policy (selects skills)
        self.high_policy = PolicyNetwork(state_dim, num_skills)
        
        # Low-level policies (execute skills)
        self.low_policies = [
            PolicyNetwork(state_dim + num_skills, 7)  # +skill embedding
            for _ in range(num_skills)
        ]
        
        # Skill embeddings
        self.skill_embeddings = nn.Embedding(num_skills, num_skills)
        
        # Optimizers
        self.high_optimizer = torch.optim.Adam(self.high_policy.parameters())
        self.low_optimizers = [
            torch.optim.Adam(policy.parameters()) 
            for policy in self.low_policies
        ]
        
    def select_skill(self, state):
        """High-level policy selects skill"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.high_policy(state_tensor)
        dist = torch.distributions.Categorical(probs)
        skill = dist.sample()
        
        return skill.item()
    
    def execute_skill(self, skill_id, state, max_steps=10):
        """Execute selected skill"""
        trajectory = []
        skill_embedding = self.skill_embeddings(torch.tensor(skill_id))
        
        for step in range(max_steps):
            # Concatenate state and skill embedding
            augmented_state = torch.cat([
                torch.FloatTensor(state),
                skill_embedding
            ]).unsqueeze(0)
            
            # Get action from low-level policy
            action_probs = self.low_policies[skill_id](augmented_state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            
            trajectory.append({
                'state': state,
                'action': action.item(),
                'skill': skill_id
            })
            
            # Check termination condition
            if self.skill_terminated(skill_id, state):
                break
        
        return trajectory
    
    def skill_terminated(self, skill_id, state):
        """Check if skill execution should terminate"""
        # Skill-specific termination conditions
        return False  # Placeholder
```

---

## 🔬 핵심 개념 정리

### 1. Exploration vs Exploitation
```python
class ExplorationStrategies:
    """다양한 탐험 전략"""
    
    @staticmethod
    def epsilon_greedy(q_values, epsilon=0.1):
        """Epsilon-greedy exploration"""
        if np.random.random() < epsilon:
            return np.random.randint(len(q_values))
        return np.argmax(q_values)
    
    @staticmethod
    def boltzmann(q_values, temperature=1.0):
        """Boltzmann exploration"""
        exp_q = np.exp(q_values / temperature)
        probs = exp_q / exp_q.sum()
        return np.random.choice(len(q_values), p=probs)
    
    @staticmethod
    def ucb(q_values, counts, c=2.0):
        """Upper Confidence Bound"""
        total_count = counts.sum()
        ucb_values = q_values + c * np.sqrt(np.log(total_count) / (counts + 1e-8))
        return np.argmax(ucb_values)
    
    @staticmethod
    def thompson_sampling(alpha, beta):
        """Thompson sampling for bandits"""
        samples = [np.random.beta(a, b) for a, b in zip(alpha, beta)]
        return np.argmax(samples)
```

### 2. Credit Assignment
```python
def monte_carlo_returns(rewards, gamma=0.99):
    """Monte Carlo returns (full episode)"""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

def td_lambda_returns(rewards, values, lambda_=0.95, gamma=0.99):
    """TD(λ) returns"""
    returns = []
    G_lambda = values[-1]
    
    for t in reversed(range(len(rewards))):
        G_lambda = rewards[t] + gamma * ((1 - lambda_) * values[t+1] + lambda_ * G_lambda)
        returns.insert(0, G_lambda)
    
    return returns
```

---

## 🛠️ 실습 코드

### 완전한 로봇 RL 시스템
```python
import torch
import torch.nn as nn
import numpy as np
import gym
from collections import deque

class CompleteRobotRL:
    """완전한 로봇 RL 시스템"""
    def __init__(self, env_name='RobotArm-v0'):
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # Multiple algorithms
        self.algorithms = {
            'dqn': DQNAgent(self.state_dim, self.action_dim),
            'reinforce': REINFORCE(self.state_dim, self.action_dim),
            'a2c': A2C(self.state_dim, self.action_dim),
            'ppo': PPO(self.state_dim, self.action_dim)
        }
        
        # Current algorithm
        self.current_algo = 'ppo'
        
        # Training statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
    def train(self, num_episodes=1000, algorithm='ppo'):
        """Train with specified algorithm"""
        self.current_algo = algorithm
        agent = self.algorithms[algorithm]
        
        for episode in range(num_episodes):
            if algorithm == 'dqn':
                reward, length = self.train_dqn_episode(agent)
            elif algorithm == 'reinforce':
                reward, length = self.train_pg_episode(agent)
            elif algorithm in ['a2c', 'ppo']:
                reward, length = self.train_ac_episode(agent)
            
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards)
                avg_length = np.mean(self.episode_lengths)
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Avg Length = {avg_length:.1f}")
    
    def train_dqn_episode(self, agent):
        """Train DQN for one episode"""
        state = self.env.reset()
        total_reward = 0
        steps = 0
        
        epsilon = max(0.01, 0.9 * (0.99 ** len(self.episode_rewards)))
        
        while True:
            # Select action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = agent.q_network.get_action(state_tensor, epsilon)
            
            # Step environment
            next_state, reward, done, _ = self.env.step(action)
            
            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Update network
            if len(agent.replay_buffer) > agent.batch_size:
                agent.update()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        return total_reward, steps
    
    def train_pg_episode(self, agent):
        """Train policy gradient for one episode"""
        trajectory = agent.collect_trajectory(self.env)
        
        # Update policy
        agent.update([trajectory])
        
        return sum(trajectory['rewards']), len(trajectory['rewards'])
    
    def train_ac_episode(self, agent):
        """Train actor-critic for one episode"""
        state = self.env.reset()
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        total_reward = 0
        steps = 0
        
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action and value
            action, log_prob, value = agent.actor_critic.get_action_and_value(state_tensor)
            
            # Step environment
            next_state, reward, done, _ = self.env.step(action)
            
            # Store transition
            trajectory['states'].append(state)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)
            trajectory['values'].append(value)
            trajectory['log_probs'].append(log_prob)
            trajectory['dones'].append(done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Update network
        agent.update([trajectory])
        
        return total_reward, steps
    
    def evaluate(self, num_episodes=10, render=False):
        """Evaluate current policy"""
        agent = self.algorithms[self.current_algo]
        
        eval_rewards = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            
            while True:
                if render:
                    self.env.render()
                
                # Get action
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                if hasattr(agent, 'policy'):
                    action = agent.policy.get_action(state_tensor)[0]
                elif hasattr(agent, 'actor_critic'):
                    action = agent.actor_critic.get_action_and_value(state_tensor)[0]
                else:
                    action = agent.q_network.get_action(state_tensor, epsilon=0)
                
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                
                if done:
                    break
            
            eval_rewards.append(total_reward)
        
        return np.mean(eval_rewards), np.std(eval_rewards)
    
    def save_model(self, path):
        """Save trained model"""
        agent = self.algorithms[self.current_algo]
        
        if hasattr(agent, 'policy'):
            model = agent.policy
        elif hasattr(agent, 'actor_critic'):
            model = agent.actor_critic
        else:
            model = agent.q_network
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'algorithm': self.current_algo,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, path)
    
    def load_model(self, path):
        """Load trained model"""
        checkpoint = torch.load(path)
        self.current_algo = checkpoint['algorithm']
        
        agent = self.algorithms[self.current_algo]
        
        if hasattr(agent, 'policy'):
            agent.policy.load_state_dict(checkpoint['model_state_dict'])
        elif hasattr(agent, 'actor_critic'):
            agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        else:
            agent.q_network.load_state_dict(checkpoint['model_state_dict'])

# Visualization utilities
def plot_training_curves(rewards, lengths):
    """Plot training progress"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Rewards
    axes[0].plot(rewards)
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    
    # Moving average
    window = 100
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(rewards)), moving_avg, 'r-', label='Moving Avg')
        axes[0].legend()
    
    # Episode lengths
    axes[1].plot(lengths)
    axes[1].set_title('Episode Lengths')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    
    plt.tight_layout()
    plt.show()

# Demo
if __name__ == "__main__":
    # Create system
    rl_system = CompleteRobotRL()
    
    # Train with different algorithms
    print("Training with PPO...")
    rl_system.train(num_episodes=500, algorithm='ppo')
    
    # Evaluate
    mean_reward, std_reward = rl_system.evaluate(num_episodes=10)
    print(f"Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Save model
    rl_system.save_model("robot_rl_model.pth")
    
    # Plot results
    plot_training_curves(
        list(rl_system.episode_rewards),
        list(rl_system.episode_lengths)
    )
```

---

## 📈 다음 단계

### 1. 고급 RL 기법
- **SAC**: Soft Actor-Critic (continuous)
- **TD3**: Twin Delayed DDPG
- **Rainbow**: DQN improvements combined

### 2. VLA 특화 개선
- **Sim2Real**: 시뮬레이션→실제 전이
- **Multi-task RL**: 다중 작업 학습
- **Meta-RL**: 빠른 적응

### 3. 최신 연구
- **Offline RL**: 데이터셋에서 학습
- **Model-based RL**: 환경 모델 활용
- **Causal RL**: 인과 관계 추론

---

## 💡 핵심 포인트

### ✅ 기억해야 할 것들
1. **Value vs Policy**: 가치 기반 vs 정책 기반
2. **On-policy vs Off-policy**: 데이터 효율성
3. **Exploration**: 충분한 탐험 필요
4. **Credit assignment**: 장기 보상 할당

### ⚠️ 주의사항
1. **Sample efficiency**: 많은 데이터 필요
2. **Stability**: 학습 불안정성
3. **Safety**: 실제 로봇에서 위험

### 🎯 VLA 적용 시
1. **Reward shaping**: 적절한 보상 설계
2. **Safety constraints**: 안전 제약
3. **Hierarchical**: 복잡한 작업 분해

---

**다음 문서**: `11_flow_models.md` - Flow 기반 생성 모델