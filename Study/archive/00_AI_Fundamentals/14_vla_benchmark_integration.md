# 🔌 VLA 모델을 벤치마크에 통합하는 완전 가이드

## 📌 개요
VLA 모델을 다양한 로봇 벤치마크에 통합하는 실전 가이드입니다. 각 벤치마크의 인터페이스에 맞춰 VLA 모델을 적용하는 방법을 상세히 다룹니다.

---

## 🏗️ 1. VLA 모델 기본 구조

### 표준 VLA 모델 인터페이스

```python
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import numpy as np

class BaseVLAModel(nn.Module):
    """모든 벤치마크에서 사용할 기본 VLA 모델"""
    
    def __init__(self, 
                 vision_encoder,
                 language_encoder,
                 action_decoder,
                 action_dim=7,
                 proprio_dim=7):
        super().__init__()
        
        # Core components
        self.vision_encoder = vision_encoder
        self.language_encoder = language_encoder
        self.action_decoder = action_decoder
        
        # Dimensions
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(768 * 2 + proprio_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512)
        )
        
        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, 
                images: torch.Tensor,
                language: torch.Tensor,
                proprioception: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] or [B, T, C, H, W] for video
            language: [B, L] tokenized instruction
            proprioception: [B, proprio_dim] robot state
        Returns:
            actions: [B, action_dim]
        """
        # Encode modalities
        visual_features = self.vision_encoder(images)
        language_features = self.language_encoder(language)
        
        # Combine features
        if proprioception is not None:
            combined = torch.cat([
                visual_features,
                language_features,
                proprioception
            ], dim=-1)
        else:
            combined = torch.cat([
                visual_features,
                language_features,
                torch.zeros(images.shape[0], self.proprio_dim).to(images.device)
            ], dim=-1)
        
        # Fuse and decode
        fused = self.fusion(combined)
        actions = self.action_head(fused)
        
        return actions
    
    def process_observation(self, obs: Dict) -> Tuple[torch.Tensor, ...]:
        """벤치마크별 관찰을 모델 입력으로 변환"""
        raise NotImplementedError
    
    def process_action(self, action: torch.Tensor, action_space) -> np.ndarray:
        """모델 출력을 벤치마크 액션으로 변환"""
        raise NotImplementedError
```

### 멀티모달 인코더 구현

```python
class VisionEncoder(nn.Module):
    """비전 인코더 - ResNet/ViT 기반"""
    def __init__(self, model_name='resnet50', pretrained=True):
        super().__init__()
        
        if 'resnet' in model_name:
            import torchvision.models as models
            backbone = getattr(models, model_name)(pretrained=pretrained)
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])
            self.output_dim = backbone.fc.in_features
        elif 'vit' in model_name:
            from transformers import ViTModel
            self.encoder = ViTModel.from_pretrained(f'google/{model_name}')
            self.output_dim = self.encoder.config.hidden_size
        
        # Projection to common dimension
        self.projection = nn.Linear(self.output_dim, 768)
    
    def forward(self, images):
        if len(images.shape) == 5:  # [B, T, C, H, W]
            B, T = images.shape[:2]
            images = images.view(B * T, *images.shape[2:])
            features = self.encoder(images)
            features = features.view(B, T, -1).mean(dim=1)  # Temporal pooling
        else:  # [B, C, H, W]
            features = self.encoder(images)
        
        return self.projection(features.squeeze())

class LanguageEncoder(nn.Module):
    """언어 인코더 - BERT/RoBERTa 기반"""
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        from transformers import AutoModel
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.output_dim = self.encoder.config.hidden_size
        self.projection = nn.Linear(self.output_dim, 768)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        return self.projection(pooled_output)
```

---

## 🎮 2. RLBench 통합

### RLBench용 VLA 래퍼

```python
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
import torch
import numpy as np

class RLBenchVLA:
    """RLBench 환경용 VLA 모델 래퍼"""
    
    def __init__(self, model: BaseVLAModel, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # RLBench specific settings
        self.image_size = (128, 128)
        self.action_mode = 'end_effector'  # or 'joint_velocity'
        
        # Tokenizer for language
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Action normalization parameters
        self.action_scale = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 1.0])
        self.action_offset = np.array([0, 0, 0, 0, 0, 0, 0])
    
    def process_observation(self, obs):
        """RLBench 관찰을 VLA 입력으로 변환"""
        
        # Extract images from multiple cameras
        images = []
        for cam in ['left_shoulder_rgb', 'right_shoulder_rgb', 'wrist_rgb']:
            if hasattr(obs, cam):
                img = getattr(obs, cam)
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                images.append(img)
        
        # Stack images or use primary camera
        if len(images) > 1:
            # Multi-view: concatenate along channel dimension
            image_tensor = torch.cat(images, dim=0)
        else:
            image_tensor = images[0]
        
        # Normalize
        image_tensor = self.normalize_image(image_tensor)
        
        # Get proprioception
        proprio = torch.from_numpy(np.concatenate([
            obs.joint_positions,
            obs.joint_velocities,
            [obs.gripper_open]
        ])).float()
        
        return image_tensor.unsqueeze(0).to(self.device), proprio.unsqueeze(0).to(self.device)
    
    def process_language(self, instruction: str):
        """언어 명령 처리"""
        tokens = self.tokenizer(
            instruction,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=64
        )
        return tokens['input_ids'].to(self.device)
    
    def predict(self, obs, instruction: str):
        """액션 예측"""
        self.model.eval()
        
        with torch.no_grad():
            # Process inputs
            images, proprio = self.process_observation(obs)
            language = self.process_language(instruction)
            
            # Forward pass
            action = self.model(images, language, proprio)
            
            # Convert to numpy and denormalize
            action = action.cpu().numpy()[0]
            action = action * self.action_scale + self.action_offset
        
        return self.format_action(action)
    
    def format_action(self, action):
        """액션 포맷팅"""
        if self.action_mode == 'end_effector':
            # [x, y, z, qx, qy, qz, qw, gripper]
            return np.concatenate([
                action[:3],  # position
                action[3:7] / np.linalg.norm(action[3:7]),  # quaternion (normalized)
                [action[6]]  # gripper
            ])
        elif self.action_mode == 'joint_velocity':
            # Direct joint velocities
            return action
        else:
            raise ValueError(f"Unknown action mode: {self.action_mode}")
    
    def normalize_image(self, image):
        """이미지 정규화"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (image - mean) / std
```

### RLBench 학습 루프

```python
class RLBenchTrainer:
    """RLBench에서 VLA 모델 학습"""
    
    def __init__(self, model, env, task_class):
        self.model = model
        self.env = env
        self.task = env.get_task(task_class)
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10000
        )
        
        # Replay buffer
        self.buffer = ReplayBuffer(capacity=10000)
        
    def collect_demonstration(self, num_demos=100):
        """전문가 시연 수집"""
        demos = []
        
        for _ in range(num_demos):
            demo = self.task.get_demos(1, live_demos=False)[0]
            
            for obs in demo:
                # Store observation-action pairs
                self.buffer.add({
                    'obs': obs,
                    'action': obs.joint_velocities,  # or end_effector_pose
                    'instruction': self.task.get_task_descriptions()[0]
                })
        
        return demos
    
    def train_step(self, batch_size=32):
        """학습 스텝"""
        batch = self.buffer.sample(batch_size)
        
        # Prepare batch
        images = []
        proprios = []
        languages = []
        actions = []
        
        for sample in batch:
            img, proprio = self.model.process_observation(sample['obs'])
            lang = self.model.process_language(sample['instruction'])
            
            images.append(img)
            proprios.append(proprio)
            languages.append(lang)
            actions.append(torch.from_numpy(sample['action']).float())
        
        images = torch.cat(images)
        proprios = torch.cat(proprios)
        languages = torch.cat(languages)
        actions = torch.stack(actions).to(self.model.device)
        
        # Forward pass
        pred_actions = self.model(images, languages, proprios)
        
        # Compute loss
        loss = F.mse_loss(pred_actions, actions)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def evaluate(self, num_episodes=10):
        """평가"""
        successes = []
        
        for _ in range(num_episodes):
            descriptions, obs = self.task.reset()
            instruction = descriptions[0]
            
            done = False
            while not done:
                action = self.model.predict(obs, instruction)
                obs, reward, done = self.task.step(action)
            
            successes.append(reward > 0)
        
        return np.mean(successes)
```

---

## 🔧 3. Meta-World 통합

### Meta-World용 VLA 어댑터

```python
import metaworld
import numpy as np

class MetaWorldVLA:
    """Meta-World 환경용 VLA 어댑터"""
    
    def __init__(self, model: BaseVLAModel, camera_name='corner'):
        self.model = model
        self.camera_name = camera_name
        
        # Meta-World doesn't have built-in language, so we create task descriptions
        self.task_descriptions = {
            'reach-v2': 'Reach the target position',
            'push-v2': 'Push the block to the goal',
            'pick-place-v2': 'Pick up the block and place it at the target',
            'door-open-v2': 'Open the door',
            'drawer-open-v2': 'Open the drawer',
            'drawer-close-v2': 'Close the drawer',
            'button-press-v2': 'Press the button',
            'peg-insert-v2': 'Insert the peg into the hole',
            'window-open-v2': 'Open the window',
            'window-close-v2': 'Close the window'
        }
        
        # Action space: [x, y, z, gripper]
        self.action_dim = 4
        
    def get_image_obs(self, env):
        """환경에서 이미지 관찰 획득"""
        # Render image from environment
        img = env.render(offscreen=True, camera_name=self.camera_name)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return img_tensor
    
    def get_proprio_obs(self, obs):
        """Proprioceptive 관찰 추출"""
        # Meta-World observation structure:
        # obs[:3] - end effector position
        # obs[3:6] - object position
        # obs[6:9] - object orientation (if applicable)
        # obs[9:] - additional task-specific features
        
        proprio = torch.from_numpy(obs[:9]).float()
        return proprio
    
    def predict(self, env, obs, task_name):
        """액션 예측"""
        self.model.eval()
        
        with torch.no_grad():
            # Get observations
            img = self.get_image_obs(env).unsqueeze(0).to(self.model.device)
            proprio = self.get_proprio_obs(obs).unsqueeze(0).to(self.model.device)
            
            # Get language instruction
            instruction = self.task_descriptions.get(task_name, 'Complete the task')
            lang_tokens = self.model.tokenizer(
                instruction,
                return_tensors='pt'
            )['input_ids'].to(self.model.device)
            
            # Predict action
            action = self.model(img, lang_tokens, proprio)
            action = action.cpu().numpy()[0]
        
        # Meta-World expects [x, y, z, gripper] where gripper is binary
        action[3] = 1 if action[3] > 0 else -1  # Binarize gripper
        
        return action
```

### Meta-World 멀티태스크 학습

```python
class MetaWorldMultiTaskTrainer:
    """Meta-World 멀티태스크 학습"""
    
    def __init__(self, model, benchmark='ML10'):
        self.model = model
        
        # Initialize benchmark
        if benchmark == 'ML10':
            self.ml = metaworld.ML10()
        elif benchmark == 'ML45':
            self.ml = metaworld.ML45()
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        
        # Create environments for each task
        self.train_envs = self._create_envs('train')
        self.test_envs = self._create_envs('test')
        
        # Training setup
        self.optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        self.buffer = {task: ReplayBuffer(5000) for task in self.train_envs}
    
    def _create_envs(self, split='train'):
        """환경 생성"""
        envs = {}
        
        classes = self.ml.train_classes if split == 'train' else self.ml.test_classes
        tasks = self.ml.train_tasks if split == 'train' else self.ml.test_tasks
        
        for name, env_cls in classes.items():
            env = env_cls()
            task = [t for t in tasks if t.env_name == name][0]
            env.set_task(task)
            envs[name] = env
        
        return envs
    
    def collect_data(self, num_episodes_per_task=10):
        """데이터 수집"""
        for task_name, env in self.train_envs.items():
            for _ in range(num_episodes_per_task):
                obs = env.reset()
                done = False
                
                while not done:
                    # Random action for initial data collection
                    action = env.action_space.sample()
                    next_obs, reward, done, info = env.step(action)
                    
                    # Store transition
                    self.buffer[task_name].add({
                        'obs': obs,
                        'action': action,
                        'reward': reward,
                        'next_obs': next_obs,
                        'done': done,
                        'task': task_name
                    })
                    
                    obs = next_obs
    
    def train_epoch(self, batch_size=64):
        """학습 에폭"""
        total_loss = 0
        num_updates = 0
        
        # Sample from each task buffer
        for task_name, buffer in self.buffer.items():
            if len(buffer) < batch_size:
                continue
            
            batch = buffer.sample(batch_size)
            loss = self.train_step(batch, task_name)
            total_loss += loss
            num_updates += 1
        
        return total_loss / max(num_updates, 1)
    
    def train_step(self, batch, task_name):
        """단일 학습 스텝"""
        # Prepare batch
        images = []
        proprios = []
        actions = []
        
        env = self.train_envs[task_name]
        
        for transition in batch:
            # Get image observation
            img = self.model.get_image_obs(env)
            proprio = self.model.get_proprio_obs(transition['obs'])
            
            images.append(img)
            proprios.append(proprio)
            actions.append(torch.from_numpy(transition['action']).float())
        
        images = torch.stack(images).to(self.model.device)
        proprios = torch.stack(proprios).to(self.model.device)
        actions = torch.stack(actions).to(self.model.device)
        
        # Language tokens for task
        instruction = self.model.task_descriptions[task_name]
        lang_tokens = self.model.tokenizer(
            [instruction] * len(batch),
            return_tensors='pt',
            padding=True
        )['input_ids'].to(self.model.device)
        
        # Forward pass
        pred_actions = self.model(images, lang_tokens, proprios)
        
        # Loss
        loss = F.mse_loss(pred_actions, actions)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, split='test', num_episodes=10):
        """평가"""
        envs = self.test_envs if split == 'test' else self.train_envs
        results = {}
        
        for task_name, env in envs.items():
            successes = []
            
            for _ in range(num_episodes):
                obs = env.reset()
                done = False
                
                while not done:
                    action = self.model.predict(env, obs, task_name)
                    obs, reward, done, info = env.step(action)
                
                successes.append(info.get('success', reward > 0))
            
            results[task_name] = np.mean(successes)
        
        return results
```

---

## 🗣️ 4. CALVIN 통합

### CALVIN용 VLA 구현

```python
from calvin_env.envs.play_table_env import PlayTableSimEnv
import hydra
from omegaconf import DictConfig

class CALVINVLA:
    """CALVIN 환경용 VLA 모델"""
    
    def __init__(self, model: BaseVLAModel, cfg_path):
        self.model = model
        
        # Load CALVIN configuration
        with hydra.initialize(config_path=cfg_path):
            self.cfg = hydra.compose(config_name="config")
        
        # Initialize environment
        self.env = PlayTableSimEnv(**self.cfg.env)
        
        # Language embedder for CALVIN instructions
        from sentence_transformers import SentenceTransformer
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def process_calvin_obs(self, obs):
        """CALVIN 관찰 처리"""
        # CALVIN provides multiple camera views
        rgb_static = obs['rgb_obs']['rgb_static']
        rgb_gripper = obs['rgb_obs']['rgb_gripper']
        
        # Robot proprioception
        robot_obs = obs['robot_obs']
        
        # Convert to tensors
        static_tensor = torch.from_numpy(rgb_static).permute(2, 0, 1).float() / 255.0
        gripper_tensor = torch.from_numpy(rgb_gripper).permute(2, 0, 1).float() / 255.0
        
        # Concatenate images (or process separately)
        images = torch.cat([static_tensor, gripper_tensor], dim=0)
        
        # Proprioception
        proprio = torch.from_numpy(robot_obs).float()
        
        return images, proprio
    
    def process_instruction_chain(self, instructions):
        """연속 명령 처리"""
        # CALVIN uses sentence embeddings
        embeddings = []
        
        for instruction in instructions:
            emb = self.sentence_encoder.encode(instruction)
            embeddings.append(torch.from_numpy(emb).float())
        
        return torch.stack(embeddings)
    
    def execute_instruction_chain(self, instructions, max_steps=1000):
        """명령 체인 실행"""
        results = []
        obs = self.env.reset()
        
        for instruction in instructions:
            success = self.execute_single_instruction(
                instruction, 
                obs, 
                max_steps=max_steps // len(instructions)
            )
            results.append(success)
            
            if not success:
                break
        
        return results
    
    def execute_single_instruction(self, instruction, initial_obs, max_steps=300):
        """단일 명령 실행"""
        obs = initial_obs
        
        for step in range(max_steps):
            # Get action from model
            action = self.predict_action(obs, instruction)
            
            # Step environment
            obs, reward, done, info = self.env.step(action)
            
            # Check if instruction completed
            if self.check_instruction_completion(instruction, obs, info):
                return True
        
        return False
    
    def predict_action(self, obs, instruction):
        """액션 예측"""
        self.model.eval()
        
        with torch.no_grad():
            # Process observation
            images, proprio = self.process_calvin_obs(obs)
            images = images.unsqueeze(0).to(self.model.device)
            proprio = proprio.unsqueeze(0).to(self.model.device)
            
            # Process language
            lang_emb = self.sentence_encoder.encode(instruction)
            lang_tensor = torch.from_numpy(lang_emb).float()
            lang_tensor = lang_tensor.unsqueeze(0).to(self.model.device)
            
            # Predict action
            action = self.model(images, lang_tensor, proprio)
            action = action.cpu().numpy()[0]
        
        return action
    
    def check_instruction_completion(self, instruction, obs, info):
        """명령 완료 확인"""
        # This would need task-specific logic
        # CALVIN provides success checking in info
        return info.get('success', False)
```

### CALVIN 평가 프로토콜

```python
class CALVINEvaluator:
    """CALVIN 벤치마크 평가"""
    
    def __init__(self, model, env, test_episodes):
        self.model = model
        self.env = env
        self.test_episodes = test_episodes
    
    def evaluate_success_rate(self):
        """SR_k 메트릭 계산 (k개 연속 성공)"""
        results = {i: [] for i in range(1, 6)}  # SR_1 to SR_5
        
        for episode in self.test_episodes:
            instructions = episode['instructions']
            
            # Reset environment
            obs = self.env.reset()
            
            consecutive_success = 0
            for instruction in instructions[:5]:  # Max 5 instructions
                success = self.model.execute_single_instruction(
                    instruction, obs, max_steps=300
                )
                
                if success:
                    consecutive_success += 1
                    # Record success at each level
                    for k in range(1, consecutive_success + 1):
                        if k <= 5:
                            results[k].append(1)
                else:
                    # Record failure for remaining levels
                    for k in range(consecutive_success + 1, 6):
                        results[k].append(0)
                    break
        
        # Calculate average success rate
        sr_metrics = {}
        for k in range(1, 6):
            if results[k]:
                sr_metrics[f'SR_{k}'] = np.mean(results[k])
            else:
                sr_metrics[f'SR_{k}'] = 0.0
        
        # Average completed length
        avg_len = sum(k * sr_metrics[f'SR_{k}'] for k in range(1, 6))
        sr_metrics['Avg_Len'] = avg_len
        
        return sr_metrics
```

---

## 🤖 5. RoboSuite 통합

### RoboSuite용 VLA 래퍼

```python
import robosuite as suite
from robosuite.wrappers import GymWrapper

class RoboSuiteVLA:
    """RoboSuite 환경용 VLA 래퍼"""
    
    def __init__(self, model: BaseVLAModel, env_name='Lift', robot='Panda'):
        self.model = model
        
        # Create environment
        self.env = suite.make(
            env_name=env_name,
            robots=robot,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs=True,
            horizon=200,
            reward_shaping=True,
            control_freq=20,
            camera_names=['agentview', 'robot0_eye_in_hand']
        )
        
        # Wrap for Gym interface
        self.env = GymWrapper(self.env)
        
        # Task descriptions for RoboSuite tasks
        self.task_descriptions = {
            'Lift': 'Lift the cube above the table',
            'Stack': 'Stack the red cube on the green cube',
            'NutAssembly': 'Place the nut on the peg',
            'PickPlace': 'Pick and place the object in the bin',
            'Door': 'Open the door',
            'Wipe': 'Wipe the table surface'
        }
    
    def process_robosuite_obs(self, obs):
        """RoboSuite 관찰 처리"""
        # Extract camera images
        agentview = obs['agentview_image']
        eye_in_hand = obs['robot0_eye_in_hand_image']
        
        # Convert to tensors
        agentview_tensor = torch.from_numpy(agentview).permute(2, 0, 1).float() / 255.0
        eye_tensor = torch.from_numpy(eye_in_hand).permute(2, 0, 1).float() / 255.0
        
        # Stack or concatenate views
        images = torch.cat([agentview_tensor, eye_tensor], dim=0)
        
        # Robot state
        robot_state = np.concatenate([
            obs['robot0_joint_pos'],
            obs['robot0_joint_vel'],
            obs['robot0_eef_pos'],
            obs['robot0_eef_quat'],
            [obs['robot0_gripper_qpos'][0]]  # Gripper state
        ])
        
        proprio = torch.from_numpy(robot_state).float()
        
        return images, proprio
    
    def predict(self, obs, task_name='Lift'):
        """액션 예측"""
        self.model.eval()
        
        with torch.no_grad():
            # Process observation
            images, proprio = self.process_robosuite_obs(obs)
            images = images.unsqueeze(0).to(self.model.device)
            proprio = proprio.unsqueeze(0).to(self.model.device)
            
            # Get task instruction
            instruction = self.task_descriptions.get(task_name, 'Complete the task')
            tokens = self.model.tokenizer(
                instruction,
                return_tensors='pt'
            )['input_ids'].to(self.model.device)
            
            # Predict action
            action = self.model(images, tokens, proprio)
            action = action.cpu().numpy()[0]
        
        return action
```

---

## 🔬 6. 통합 학습 파이프라인

### 범용 VLA 학습기

```python
class UniversalVLATrainer:
    """모든 벤치마크에서 사용 가능한 범용 학습기"""
    
    def __init__(self, model, benchmark_name, config):
        self.model = model
        self.benchmark_name = benchmark_name
        self.config = config
        
        # Initialize appropriate wrapper
        if benchmark_name == 'rlbench':
            self.wrapper = RLBenchVLA(model)
        elif benchmark_name == 'metaworld':
            self.wrapper = MetaWorldVLA(model)
        elif benchmark_name == 'calvin':
            self.wrapper = CALVINVLA(model, config.calvin_cfg_path)
        elif benchmark_name == 'robosuite':
            self.wrapper = RoboSuiteVLA(model)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        # Training setup
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = self.get_scheduler(config.scheduler_type)
        
        # Logging
        self.writer = SummaryWriter(config.log_dir)
        self.checkpoint_dir = config.checkpoint_dir
        
    def get_scheduler(self, scheduler_type):
        """스케줄러 선택"""
        if scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.max_steps
            )
        elif scheduler_type == 'linear':
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer, 
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config.max_steps
            )
        else:
            return None
    
    def train(self, num_epochs):
        """통합 학습 루프"""
        for epoch in range(num_epochs):
            # Collect data
            train_data = self.collect_data()
            
            # Training
            epoch_loss = 0
            for batch in self.create_batches(train_data):
                loss = self.train_step(batch)
                epoch_loss += loss
            
            # Evaluation
            if epoch % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                self.log_metrics(eval_metrics, epoch)
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(epoch)
            
            print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")
    
    def collect_data(self):
        """벤치마크별 데이터 수집"""
        if self.benchmark_name == 'rlbench':
            return self.wrapper.collect_demonstration()
        elif self.benchmark_name == 'metaworld':
            return self.wrapper.collect_random_data()
        # ... etc
    
    def train_step(self, batch):
        """통합 학습 스텝"""
        self.model.train()
        
        # Forward pass
        loss = self.wrapper.compute_loss(batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
        
        return loss.item()
    
    def evaluate(self):
        """통합 평가"""
        self.model.eval()
        return self.wrapper.evaluate()
    
    def save_checkpoint(self, epoch):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config
        }
        
        path = os.path.join(self.checkpoint_dir, f'checkpoint_{epoch}.pt')
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """체크포인트 로드"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']
```

---

## 📊 7. 성능 모니터링 및 분석

### 통합 평가 시스템

```python
class BenchmarkEvaluator:
    """모든 벤치마크 통합 평가"""
    
    def __init__(self, model, benchmarks):
        self.model = model
        self.benchmarks = benchmarks
        self.results = {}
    
    def evaluate_all(self):
        """모든 벤치마크 평가"""
        for benchmark_name in self.benchmarks:
            print(f"Evaluating on {benchmark_name}...")
            
            if benchmark_name == 'rlbench':
                results = self.evaluate_rlbench()
            elif benchmark_name == 'metaworld':
                results = self.evaluate_metaworld()
            elif benchmark_name == 'calvin':
                results = self.evaluate_calvin()
            elif benchmark_name == 'robosuite':
                results = self.evaluate_robosuite()
            
            self.results[benchmark_name] = results
        
        return self.results
    
    def create_report(self):
        """평가 리포트 생성"""
        report = {
            'summary': {},
            'detailed': self.results,
            'plots': {}
        }
        
        # Summary statistics
        for benchmark, results in self.results.items():
            report['summary'][benchmark] = {
                'success_rate': results.get('success_rate', 0),
                'average_reward': results.get('average_reward', 0),
                'completion_time': results.get('completion_time', 0)
            }
        
        # Generate plots
        report['plots'] = self.generate_plots()
        
        return report
    
    def generate_plots(self):
        """시각화 생성"""
        import matplotlib.pyplot as plt
        
        plots = {}
        
        # Success rate comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        benchmarks = list(self.results.keys())
        success_rates = [self.results[b].get('success_rate', 0) for b in benchmarks]
        
        ax.bar(benchmarks, success_rates)
        ax.set_ylabel('Success Rate')
        ax.set_title('VLA Performance Across Benchmarks')
        
        plt.tight_layout()
        plots['success_comparison'] = fig
        
        return plots
```

---

## 💡 실전 팁

### 1. 시작하기

```python
# Quick start example
model = BaseVLAModel(
    vision_encoder=VisionEncoder('resnet50'),
    language_encoder=LanguageEncoder('bert-base-uncased'),
    action_decoder=ActionDecoder()
)

# Choose benchmark
wrapper = RLBenchVLA(model)  # or MetaWorldVLA, etc.

# Train
trainer = UniversalVLATrainer(model, 'rlbench', config)
trainer.train(num_epochs=100)

# Evaluate
evaluator = BenchmarkEvaluator(model, ['rlbench'])
results = evaluator.evaluate_all()
```

### 2. 일반적인 문제 해결

1. **액션 공간 불일치**: 정규화/비정규화 확인
2. **관찰 형식 차이**: 전처리 파이프라인 검증
3. **언어 인터페이스**: 벤치마크별 적절한 명령 생성
4. **성능 최적화**: 배치 처리, GPU 활용

### 3. 최적화 전략

- **Multi-GPU Training**: DistributedDataParallel 사용
- **Mixed Precision**: AMP로 속도 향상
- **Efficient Sampling**: 병렬 환경 실행
- **Curriculum Learning**: 쉬운 작업부터 시작

---

## 📚 추가 자료

### 코드 저장소
- [VLA-Benchmark-Integration](https://github.com/example/vla-benchmark)
- [Universal-Robot-Learning](https://github.com/example/url)

### 튜토리얼
- RLBench + VLA 통합 노트북
- Meta-World 멀티태스크 학습 가이드
- CALVIN 언어 조건부 제어 예제

---

## 🎯 핵심 요약

VLA 모델을 벤치마크에 통합하려면 각 벤치마크의 관찰/액션 공간을 이해하고 적절한 래퍼를 구현해야 합니다. 통합 과정은 1) 관찰 전처리, 2) 언어 명령 처리, 3) 액션 후처리, 4) 평가 메트릭 계산으로 구성됩니다. 범용 학습 파이프라인을 구축하면 여러 벤치마크에서 일관된 방식으로 모델을 학습하고 평가할 수 있습니다.