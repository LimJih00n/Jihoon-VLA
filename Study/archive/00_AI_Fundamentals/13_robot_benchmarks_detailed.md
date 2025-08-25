# 🏆 Robot Benchmarks 상세 가이드

## 📌 개요
로봇 학습 연구에서 표준화된 벤치마크는 알고리즘의 성능을 객관적으로 평가하고 비교하는 데 필수적입니다. 이 문서에서는 VLA 연구에 중요한 주요 로봇 벤치마크들을 상세히 분석합니다.

---

## 🎮 1. RLBench (Robot Learning Benchmark)

### 개요
- **개발**: Imperial College London
- **작업 수**: 100+ 다양한 로봇 작업
- **시뮬레이터**: CoppeliaSim (V-REP)
- **특징**: 비전 기반, 자연어 지시, 다양한 난이도

### 주요 작업 카테고리

#### Manipulation Tasks
```python
manipulation_tasks = {
    "pick_and_place": "물체를 집어서 목표 위치에 놓기",
    "stack_blocks": "블록을 순서대로 쌓기",
    "insert_peg": "페그를 구멍에 삽입",
    "open_drawer": "서랍 열기",
    "close_jar": "병뚜껑 닫기"
}
```

#### Tool Use Tasks
```python
tool_tasks = {
    "use_hammer": "망치로 못 박기",
    "screw_nail": "나사 조이기",
    "sweep_dust": "먼지 쓸기",
    "pour_water": "물 따르기"
}
```

#### Multi-Step Tasks
```python
complex_tasks = {
    "make_coffee": "커피 만들기 (10+ 단계)",
    "set_table": "식탁 차리기",
    "sort_objects": "물체 분류 및 정리",
    "assembly": "부품 조립"
}
```

### 평가 메트릭

```python
class RLBenchMetrics:
    def __init__(self):
        self.metrics = {
            'success_rate': 0.0,      # 작업 성공률
            'completion_time': 0.0,    # 평균 완료 시간
            'path_efficiency': 0.0,    # 경로 효율성
            'smoothness': 0.0,         # 동작 부드러움
            'generalization': 0.0      # 일반화 성능
        }
    
    def evaluate_episode(self, trajectory):
        """에피소드 평가"""
        success = self.check_task_success(trajectory)
        time = len(trajectory)
        efficiency = self.compute_path_efficiency(trajectory)
        smoothness = self.compute_smoothness(trajectory)
        
        return {
            'success': success,
            'time': time,
            'efficiency': efficiency,
            'smoothness': smoothness
        }
    
    def compute_path_efficiency(self, trajectory):
        """최적 경로 대비 효율성"""
        optimal_path = self.get_optimal_path(trajectory[0], trajectory[-1])
        actual_length = sum(np.linalg.norm(trajectory[i+1] - trajectory[i]) 
                           for i in range(len(trajectory)-1))
        return optimal_path / actual_length
    
    def compute_smoothness(self, trajectory):
        """가속도 변화 측정"""
        accelerations = np.diff(trajectory, n=2, axis=0)
        jerk = np.diff(accelerations, axis=0)
        return 1.0 / (1.0 + np.mean(np.abs(jerk)))
```

### 데이터 형식

```python
class RLBenchObservation:
    def __init__(self):
        self.left_shoulder_rgb = None     # (128, 128, 3)
        self.right_shoulder_rgb = None    # (128, 128, 3)
        self.overhead_rgb = None          # (128, 128, 3)
        self.wrist_rgb = None            # (128, 128, 3)
        self.left_shoulder_depth = None   # (128, 128)
        self.right_shoulder_depth = None  # (128, 128)
        self.overhead_depth = None        # (128, 128)
        self.wrist_depth = None          # (128, 128)
        self.joint_positions = None       # (7,) for 7-DOF arm
        self.joint_velocities = None      # (7,)
        self.gripper_open = None         # float [0, 1]
        self.task_low_dim_state = None   # Task-specific state

class RLBenchAction:
    def __init__(self):
        self.joint_positions = None       # (7,) target positions
        self.gripper_open = None         # float [0, 1]
        # OR
        self.end_effector_pose = None    # (7,) position + quaternion
        self.gripper_action = None       # binary open/close
```

### 사용 예시

```python
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

# 환경 설정
obs_config = ObservationConfig()
obs_config.set_all(True)  # 모든 카메라 활성화

action_mode = ActionMode(
    arm=ArmActionMode.ABS_JOINT_VELOCITY,  # 절대 관절 속도 제어
    gripper=GripperActionMode.OPEN_CLOSE    # 그리퍼 열기/닫기
)

env = Environment(
    action_mode=action_mode,
    obs_config=obs_config,
    headless=True  # GUI 없이 실행
)

# 작업 로드
task = env.get_task(ReachTarget)
descriptions, obs = task.reset()

# 에피소드 실행
done = False
while not done:
    action = agent.predict(obs)  # VLA 모델 예측
    obs, reward, done = task.step(action)

env.shutdown()
```

---

## 🔧 2. Meta-World

### 개요
- **개발**: UC Berkeley
- **작업 수**: 50개 표준화된 조작 작업
- **시뮬레이터**: MuJoCo
- **특징**: 멀티태스크, 메타러닝, 연속 제어

### 작업 분류

#### ML1 (Single Task)
```python
ml1_tasks = [
    'reach-v2',          # 목표 지점 도달
    'push-v2',           # 물체 밀기
    'pick-place-v2',     # 집어서 놓기
    'door-open-v2',      # 문 열기
    'drawer-open-v2',    # 서랍 열기
    'drawer-close-v2',   # 서랍 닫기
    'button-press-v2',   # 버튼 누르기
    'peg-insert-v2',     # 페그 삽입
    'window-open-v2',    # 창문 열기
    'window-close-v2'    # 창문 닫기
]
```

#### ML10 (10 Tasks)
```python
ml10_train = [
    'reach-v2', 'push-v2', 'pick-place-v2', 
    'door-open-v2', 'drawer-close-v2'
]

ml10_test = [
    'drawer-open-v2', 'door-close-v2', 
    'shelf-place-v2', 'sweep-into-v2', 'lever-pull-v2'
]
```

#### ML45 (45 Tasks)
```python
# 45개 작업 모두 포함 (ML10 제외)
ml45_tasks = metaworld.ML45().train_classes
```

### 환경 인터페이스

```python
import metaworld
import random

class MetaWorldEnvironment:
    def __init__(self, benchmark_name='ML1'):
        if benchmark_name == 'ML1':
            self.ml = metaworld.ML1('pick-place-v2')
        elif benchmark_name == 'ML10':
            self.ml = metaworld.ML10()
        elif benchmark_name == 'ML45':
            self.ml = metaworld.ML45()
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        self.training_envs = []
        for name, env_cls in self.ml.train_classes.items():
            env = env_cls()
            task = random.choice([task for task in self.ml.train_tasks
                                 if task.env_name == name])
            env.set_task(task)
            self.training_envs.append(env)
    
    def reset(self, task_idx=0):
        """특정 작업 리셋"""
        obs = self.training_envs[task_idx].reset()
        return self.process_observation(obs)
    
    def step(self, action, task_idx=0):
        """액션 실행"""
        obs, reward, done, info = self.training_envs[task_idx].step(action)
        return self.process_observation(obs), reward, done, info
    
    def process_observation(self, obs):
        """관찰 전처리"""
        return {
            'robot_state': obs[:9],      # 로봇 상태 (위치, 그리퍼)
            'object_state': obs[9:],     # 물체 상태
            'full_state': obs            # 전체 상태
        }
```

### 성능 평가

```python
class MetaWorldEvaluator:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
    
    def evaluate(self, num_episodes=50):
        """정책 평가"""
        successes = []
        rewards = []
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.policy.get_action(obs)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
            
            successes.append(info['success'])
            rewards.append(episode_reward)
        
        return {
            'success_rate': np.mean(successes),
            'average_reward': np.mean(rewards),
            'std_reward': np.std(rewards)
        }
    
    def evaluate_multitask(self, tasks):
        """멀티태스크 평가"""
        results = {}
        for task_name, task_env in tasks.items():
            self.env = task_env
            results[task_name] = self.evaluate()
        
        # 전체 성능
        overall_success = np.mean([r['success_rate'] for r in results.values()])
        return results, overall_success
```

---

## 🗣️ 3. CALVIN (Composing Actions from Language and Vision)

### 개요
- **개발**: University of Freiburg
- **특징**: 언어 조건부, 장기 작업, 조합적 일반화
- **시뮬레이터**: PyBullet
- **데이터**: 24시간 텔레오퍼레이션 데이터

### 작업 구조

```python
class CALVINTask:
    def __init__(self):
        self.atomic_tasks = {
            'pick': 'Pick up the {object}',
            'place': 'Place it on the {location}',
            'push': 'Push the {object} to the {location}',
            'pull': 'Pull the {object}',
            'open': 'Open the {container}',
            'close': 'Close the {container}',
            'rotate': 'Rotate the {object}',
            'stack': 'Stack {object1} on {object2}'
        }
        
        self.composite_tasks = [
            'Pick up the red block and place it in the drawer',
            'Open the drawer, then close it',
            'Stack all blocks on the table',
            'Sort objects by color',
            'Move all objects to the left side'
        ]
```

### 언어 조건부 제어

```python
class CALVINLanguageConditionedPolicy:
    def __init__(self, vision_encoder, language_encoder, policy_network):
        self.vision_encoder = vision_encoder
        self.language_encoder = language_encoder
        self.policy_network = policy_network
    
    def process_instruction(self, instruction):
        """언어 명령 처리"""
        # Tokenize and encode
        tokens = self.tokenize(instruction)
        language_embedding = self.language_encoder(tokens)
        return language_embedding
    
    def predict_action(self, observation, instruction):
        """언어 조건부 행동 예측"""
        # Vision encoding
        visual_features = self.vision_encoder(observation['rgb'])
        
        # Language encoding
        language_features = self.process_instruction(instruction)
        
        # Fuse modalities
        fused_features = torch.cat([visual_features, language_features], dim=-1)
        
        # Predict action
        action = self.policy_network(fused_features)
        
        return action
    
    def execute_sequence(self, observations, instruction_sequence):
        """연속 명령 실행"""
        actions = []
        for obs, instruction in zip(observations, instruction_sequence):
            action = self.predict_action(obs, instruction)
            actions.append(action)
        return actions
```

### 평가 프로토콜

```python
class CALVINEvaluator:
    def __init__(self):
        self.evaluation_splits = {
            'D': 'Same environment',           # 같은 환경
            'ABC': 'Different environments',   # 다른 환경
            'ABCD': 'All environments'        # 모든 환경
        }
    
    def evaluate_success_rate(self, policy, test_episodes):
        """성공률 평가"""
        results = {
            1: 0,  # 1개 명령 성공
            2: 0,  # 2개 연속 성공
            3: 0,  # 3개 연속 성공
            4: 0,  # 4개 연속 성공
            5: 0   # 5개 연속 성공
        }
        
        for episode in test_episodes:
            consecutive_success = 0
            for instruction in episode['instructions']:
                success = self.execute_and_check(
                    policy, 
                    episode['initial_state'],
                    instruction
                )
                if success:
                    consecutive_success += 1
                    results[consecutive_success] += 1
                else:
                    break
        
        # Normalize
        for k in results:
            results[k] /= len(test_episodes)
        
        return results
    
    def compute_average_length(self, results):
        """평균 성공 길이"""
        total = sum(k * v for k, v in results.items())
        return total
```

---

## 🤖 4. RoboSuite

### 개요
- **개발**: Stanford IRIS Lab
- **시뮬레이터**: MuJoCo
- **특징**: 모듈화, 다양한 로봇, 표준화된 API

### 로봇 모델

```python
robot_models = {
    'Panda': 'Franka Emika Panda (7-DOF)',
    'Sawyer': 'Rethink Sawyer (7-DOF)',
    'IIWA': 'KUKA LBR IIWA (7-DOF)',
    'Jaco': 'Kinova Jaco (6-DOF)',
    'UR5e': 'Universal Robots UR5e (6-DOF)',
    'Baxter': 'Rethink Baxter (dual 7-DOF)'
}
```

### 작업 환경

```python
from robosuite import make
from robosuite.wrappers import GymWrapper

class RoboSuiteEnvironment:
    def __init__(self, task_name='Lift', robot='Panda'):
        self.env = make(
            env_name=task_name,
            robots=robot,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            horizon=200,
            reward_shaping=True,
            control_freq=20
        )
        
        # Gym wrapper for standard interface
        self.env = GymWrapper(self.env)
    
    def get_observation_spec(self):
        """관찰 사양"""
        return {
            'robot_state': {
                'joint_pos': (7,),
                'joint_vel': (7,),
                'eef_pos': (3,),
                'eef_quat': (4,),
                'gripper_qpos': (2,)
            },
            'object_state': {
                'object_pos': (3,),
                'object_quat': (4,)
            },
            'image_obs': {
                'agentview_image': (84, 84, 3),
                'robot0_eye_in_hand': (84, 84, 3)
            }
        }
    
    def create_controller(self, controller_type='OSC_POSE'):
        """컨트롤러 생성"""
        from robosuite.controllers import load_controller_config
        
        controller_config = load_controller_config(
            default_controller=controller_type
        )
        
        return controller_config
```

### 커스텀 작업 생성

```python
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.tasks import ManipulationTask

class CustomTask(SingleArmEnv):
    def __init__(self, robots, env_configuration="default", **kwargs):
        # Task 정의
        self.task = ManipulationTask(
            mujoco_arena=self._get_arena(),
            mujoco_robots=robots,
            mujoco_objects=self._get_objects()
        )
        
        super().__init__(robots, env_configuration, **kwargs)
    
    def _get_arena(self):
        """작업 공간 정의"""
        from robosuite.models.arenas import TableArena
        return TableArena(
            table_full_size=(0.8, 0.8, 0.05),
            table_offset=(0, 0, 0.8)
        )
    
    def _get_objects(self):
        """조작 물체 정의"""
        from robosuite.models.objects import BoxObject
        return [
            BoxObject(
                name="cube",
                size=[0.02, 0.02, 0.02],
                rgba=[1, 0, 0, 1]
            )
        ]
    
    def _get_reward(self):
        """보상 함수"""
        reward = 0
        
        # Distance reward
        dist = np.linalg.norm(self.eef_pos - self.object_pos)
        reward -= dist
        
        # Success reward
        if self._check_success():
            reward += 10
        
        return reward
    
    def _check_success(self):
        """성공 조건"""
        return self.object_height > 0.1  # 물체가 들린 경우
```

---

## 📊 5. 벤치마크 비교

### 특징 비교

| 벤치마크 | 작업 수 | 시뮬레이터 | 주요 특징 | 난이도 |
|---------|--------|-----------|----------|--------|
| RLBench | 100+ | CoppeliaSim | 비전 중심, 다양성 | 높음 |
| Meta-World | 50 | MuJoCo | 표준화, 메타러닝 | 중간 |
| CALVIN | 34 | PyBullet | 언어 조건부 | 높음 |
| RoboSuite | 9 | MuJoCo | 모듈화, 확장성 | 낮음-중간 |

### 선택 가이드

```python
def select_benchmark(requirements):
    """요구사항에 따른 벤치마크 선택"""
    
    if requirements['vision_based'] and requirements['diverse_tasks']:
        return 'RLBench'
    
    elif requirements['meta_learning'] or requirements['multi_task']:
        return 'Meta-World'
    
    elif requirements['language_conditioned']:
        return 'CALVIN'
    
    elif requirements['customizable'] and requirements['multiple_robots']:
        return 'RoboSuite'
    
    else:
        return 'Start with RoboSuite for simplicity'
```

---

## 🔬 6. 성능 측정 표준

### 공통 메트릭

```python
class StandardMetrics:
    def __init__(self):
        self.metrics = {
            'success_rate': self.compute_success_rate,
            'sample_efficiency': self.compute_sample_efficiency,
            'generalization': self.compute_generalization,
            'robustness': self.compute_robustness,
            'smoothness': self.compute_smoothness
        }
    
    def compute_success_rate(self, episodes):
        """작업 성공률"""
        successes = [ep['success'] for ep in episodes]
        return np.mean(successes)
    
    def compute_sample_efficiency(self, learning_curve):
        """샘플 효율성 (AUC)"""
        return np.trapz(learning_curve['success_rate'], 
                       learning_curve['steps'])
    
    def compute_generalization(self, train_perf, test_perf):
        """일반화 성능"""
        return test_perf / train_perf if train_perf > 0 else 0
    
    def compute_robustness(self, perturbed_episodes):
        """노이즈 강건성"""
        base_perf = self.compute_success_rate(perturbed_episodes[0])
        perturbed_perfs = [self.compute_success_rate(eps) 
                          for eps in perturbed_episodes[1:]]
        return np.mean(perturbed_perfs) / base_perf
    
    def compute_smoothness(self, trajectories):
        """궤적 부드러움"""
        smoothness_scores = []
        for traj in trajectories:
            # Compute jerk (3rd derivative)
            vel = np.diff(traj, axis=0)
            acc = np.diff(vel, axis=0)
            jerk = np.diff(acc, axis=0)
            smoothness = 1.0 / (1.0 + np.mean(np.abs(jerk)))
            smoothness_scores.append(smoothness)
        return np.mean(smoothness_scores)
```

### 리더보드 형식

```python
def create_leaderboard(results):
    """표준 리더보드 생성"""
    leaderboard = []
    
    for method_name, method_results in results.items():
        entry = {
            'Method': method_name,
            'Success Rate': f"{method_results['success_rate']:.1%}",
            'Sample Efficiency': f"{method_results['sample_efficiency']:.0f}",
            'Generalization': f"{method_results['generalization']:.2f}",
            'Robustness': f"{method_results['robustness']:.2f}",
            'Overall Score': compute_overall_score(method_results)
        }
        leaderboard.append(entry)
    
    # Sort by overall score
    leaderboard.sort(key=lambda x: x['Overall Score'], reverse=True)
    
    return pd.DataFrame(leaderboard)

def compute_overall_score(results, weights=None):
    """종합 점수 계산"""
    if weights is None:
        weights = {
            'success_rate': 0.4,
            'sample_efficiency': 0.2,
            'generalization': 0.2,
            'robustness': 0.2
        }
    
    score = sum(results[metric] * weight 
               for metric, weight in weights.items())
    return score
```

---

## 💡 실전 활용 팁

### 1. 벤치마크 시작하기

```python
# 초보자: RoboSuite부터
# - 간단한 API
# - 좋은 문서화
# - 쉬운 커스터마이징

# 중급자: Meta-World
# - 표준화된 작업
# - 멀티태스크 학습
# - 빠른 실행

# 고급자: RLBench or CALVIN
# - 복잡한 작업
# - 비전/언어 통합
# - 실제 적용 가능
```

### 2. 성능 향상 전략

1. **Curriculum Learning**: 쉬운 작업부터 시작
2. **Multi-Task Learning**: 관련 작업 동시 학습
3. **Data Augmentation**: 시각/물리 증강
4. **Sim-to-Real**: 도메인 랜덤화

### 3. 일반적인 함정 피하기

- **과적합**: 특정 작업에만 최적화
- **불공정 비교**: 다른 설정/하이퍼파라미터
- **체리피킹**: 좋은 결과만 보고
- **재현성**: 시드 고정, 환경 버전 명시

---

## 📚 추가 자료

### 논문
- RLBench: "RLBench: The Robot Learning Benchmark & Learning Environment"
- Meta-World: "Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning"
- CALVIN: "CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks"
- RoboSuite: "RoboSuite: A Modular Simulation Framework and Benchmark for Robot Learning"

### 튜토리얼
- [RLBench 공식 튜토리얼](https://github.com/stepjam/RLBench)
- [Meta-World 문서](https://meta-world.github.io/)
- [CALVIN 가이드](https://github.com/mees/calvin)
- [RoboSuite 문서](https://robosuite.ai/)

### 실습 코드
- 각 벤치마크별 baseline 구현
- VLA 모델 통합 예제
- 평가 스크립트 템플릿

---

## 🎯 핵심 요약

로봇 벤치마크는 VLA 연구의 객관적 평가를 위한 필수 도구입니다. RLBench는 비전 기반 다양한 작업, Meta-World는 표준화된 조작 작업, CALVIN은 언어 조건부 장기 작업, RoboSuite는 커스터마이징이 특징입니다. 각 벤치마크의 특성을 이해하고 연구 목적에 맞게 선택하는 것이 중요하며, 표준화된 평가 메트릭을 사용하여 공정한 비교를 수행해야 합니다.