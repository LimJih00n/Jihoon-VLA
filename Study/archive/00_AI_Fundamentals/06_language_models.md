# 🗣️ Language Models: 자연어 명령 이해

**목표**: GPT, BERT 구조 이해 및 로봇 명령어 처리를 위한 언어 모델 구현  
**시간**: 3-4시간  
**전제조건**: 01_neural_networks_basics.md, 02_attention_mechanism.md, 03_transformer_architecture.md  

---

## 🎯 개발자를 위한 직관적 이해

### Language Model의 두 가지 패러다임
```python
language_model_types = {
    "autoregressive": {
        "대표": "GPT",
        "방식": "다음 토큰 예측 (left-to-right)",
        "장점": "생성 태스크에 강함",
        "예시": "Pick up the → [red/blue/green]"
    },
    "masked": {
        "대표": "BERT", 
        "방식": "마스크된 토큰 예측 (bidirectional)",
        "장점": "이해 태스크에 강함",
        "예시": "Pick up the [MASK] ball → [red]"
    }
}

# VLA에서의 활용
vla_language_needs = {
    "instruction_understanding": "BERT 스타일이 유리",
    "action_generation": "GPT 스타일이 유리",
    "hybrid": "두 방식 결합이 최적"
}
```

---

## 🏗️ 기본 구조 및 구현

### 1. GPT-style Autoregressive Model
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPTModel(nn.Module):
    """GPT 스타일 autoregressive 언어 모델"""
    def __init__(self,
                 vocab_size=50000,
                 d_model=768,
                 n_heads=12,
                 n_layers=12,
                 d_ff=3072,
                 max_seq_len=1024,
                 dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying (share embeddings with output)
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Position embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        x = token_embeds + position_embeds
        x = self.dropout(x)
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len).to(x.device)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask)
        
        # Final layer norm and output projection
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        return logits
    
    def create_causal_mask(self, seq_len):
        """Causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=50):
        """Autoregressive text generation"""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                logits = self.forward(input_ids)
                
                # Get logits for last position
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if EOS token
                if next_token.item() == 2:  # Assuming 2 is EOS
                    break
        
        return input_ids

class TransformerBlock(nn.Module):
    """Single transformer block for GPT"""
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        # Self-attention
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.ln1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        
        return x
```

### 2. BERT-style Masked Language Model
```python
class BERTModel(nn.Module):
    """BERT 스타일 bidirectional 언어 모델"""
    def __init__(self,
                 vocab_size=50000,
                 d_model=768,
                 n_heads=12,
                 n_layers=12,
                 d_ff=3072,
                 max_seq_len=512,
                 dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)  # For sentence pairs
        
        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.dropout = nn.Dropout(dropout)
        self.embedding_norm = nn.LayerNorm(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
        
        # NSP/Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 2)
        )
        
    def forward(self, input_ids, attention_mask=None, segment_ids=None, masked_positions=None):
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        
        # Add segment embeddings if provided
        if segment_ids is not None:
            segment_embeds = self.segment_embedding(segment_ids)
            embeddings = token_embeds + position_embeds + segment_embeds
        else:
            embeddings = token_embeds + position_embeds
        
        # Apply masking if specified
        if masked_positions is not None:
            mask_embeds = self.mask_token.expand(batch_size, -1, -1)
            embeddings[masked_positions] = mask_embeds.squeeze(1)
        
        # Normalize and dropout
        embeddings = self.embedding_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Transformer encoding
        encoded = self.encoder(embeddings, src_key_padding_mask=attention_mask)
        
        # MLM predictions
        mlm_logits = self.mlm_head(encoded)
        
        # Classification from CLS token
        cls_logits = self.cls_head(encoded[:, 0])
        
        return {
            'mlm_logits': mlm_logits,
            'cls_logits': cls_logits,
            'hidden_states': encoded
        }
    
    def mask_tokens(self, input_ids, mlm_probability=0.15):
        """랜덤하게 토큰 마스킹"""
        labels = input_ids.clone()
        
        # Create probability matrix
        probability_matrix = torch.full(input_ids.shape, mlm_probability)
        
        # Don't mask special tokens (0: PAD, 1: CLS, 2: SEP)
        special_tokens_mask = (input_ids <= 2)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Create mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens
        
        # 80% of time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = 103  # [MASK] token id
        
        # 10% of time, replace with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.token_embedding.weight), input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        # 10% of time, keep original
        
        return input_ids, labels, masked_indices
```

### 3. Instruction-Following Language Model
```python
class InstructionLM(nn.Module):
    """로봇 명령어 이해를 위한 언어 모델"""
    def __init__(self, base_model='bert', vocab_size=50000, d_model=768):
        super().__init__()
        
        # Base language model
        if base_model == 'bert':
            self.lm = BERTModel(vocab_size=vocab_size, d_model=d_model)
            self.use_cls = True
        else:
            self.lm = GPTModel(vocab_size=vocab_size, d_model=d_model)
            self.use_cls = False
        
        # Instruction understanding head
        self.instruction_encoder = nn.LSTM(
            d_model, d_model // 2, 
            num_layers=2, bidirectional=True, batch_first=True
        )
        
        # Action grounding head
        self.action_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 256)
        )
        
        # Semantic role labeling
        self.srl_head = nn.Linear(d_model, 20)  # 20 semantic roles
        
        # Intent classification
        self.intent_classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 10 intent classes
        )
        
    def forward(self, input_ids, attention_mask=None):
        # Get language model outputs
        if self.use_cls:
            outputs = self.lm(input_ids, attention_mask)
            hidden_states = outputs['hidden_states']
        else:
            logits = self.lm(input_ids, attention_mask)
            hidden_states = logits  # Use last layer outputs
        
        # Process through LSTM for sequential understanding
        lstm_out, (hidden, cell) = self.instruction_encoder(hidden_states)
        
        # Get instruction representation
        if self.use_cls:
            instruction_repr = hidden_states[:, 0]  # CLS token
        else:
            instruction_repr = lstm_out[:, -1]  # Last position
        
        # Decode to action space
        action_features = self.action_decoder(instruction_repr)
        
        # Semantic role labeling for each token
        srl_logits = self.srl_head(hidden_states)
        
        # Intent classification
        intent_logits = self.intent_classifier(instruction_repr)
        
        return {
            'action_features': action_features,
            'srl_logits': srl_logits,
            'intent_logits': intent_logits,
            'hidden_states': hidden_states
        }
    
    def parse_instruction(self, instruction_text, tokenizer):
        """명령어 파싱 및 구조화"""
        # Tokenize
        tokens = tokenizer.encode(instruction_text)
        input_ids = torch.tensor([tokens])
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(input_ids)
        
        # Extract components
        intent = torch.argmax(outputs['intent_logits'], dim=-1)
        srl_tags = torch.argmax(outputs['srl_logits'], dim=-1)
        
        # Parse into structured format
        parsed = {
            'intent': self.intent_to_string(intent.item()),
            'tokens': tokenizer.convert_ids_to_tokens(tokens),
            'semantic_roles': self.extract_semantic_roles(tokens, srl_tags[0]),
            'action_embedding': outputs['action_features']
        }
        
        return parsed
    
    def intent_to_string(self, intent_id):
        """Intent ID를 문자열로 변환"""
        intents = ['pick', 'place', 'move', 'push', 'pull', 
                  'rotate', 'open', 'close', 'pour', 'stack']
        return intents[intent_id] if intent_id < len(intents) else 'unknown'
    
    def extract_semantic_roles(self, tokens, srl_tags):
        """Semantic role 추출"""
        roles = {
            0: 'action',
            1: 'object',
            2: 'location',
            3: 'direction',
            4: 'manner',
            5: 'tool'
        }
        
        result = {}
        for i, (token, tag) in enumerate(zip(tokens, srl_tags)):
            tag_id = tag.item()
            if tag_id in roles:
                role = roles[tag_id]
                if role not in result:
                    result[role] = []
                result[role].append(token)
        
        return result
```

### 4. Efficient Language Models for Robotics
```python
class EfficientRobotLM(nn.Module):
    """로봇을 위한 경량 언어 모델"""
    def __init__(self, vocab_size=10000, d_model=256, n_layers=4):
        super().__init__()
        
        # Smaller embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Efficient transformer (fewer layers, heads)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead=4, dim_feedforward=512,
                dropout=0.1, batch_first=True
            ),
            num_layers=n_layers
        )
        
        # Task-specific heads
        self.heads = nn.ModuleDict({
            'object_detection': nn.Linear(d_model, 100),  # 100 object classes
            'spatial_relation': nn.Linear(d_model, 20),   # 20 spatial relations
            'action_type': nn.Linear(d_model, 15),        # 15 action types
            'modifiers': nn.Linear(d_model, 30)           # 30 modifiers
        })
        
        # Cache for faster inference
        self.cache_enabled = False
        self.kv_cache = {}
        
    def forward(self, input_ids, use_cache=False):
        # Embedding
        x = self.embedding(input_ids)
        
        # Use cache if enabled
        if use_cache and self.cache_enabled:
            # Retrieve from cache
            if 'hidden' in self.kv_cache:
                # Concatenate with cached
                cached = self.kv_cache['hidden']
                x = torch.cat([cached, x], dim=1)
        
        # Transformer encoding
        encoded = self.transformer(x)
        
        # Update cache
        if use_cache:
            self.kv_cache['hidden'] = encoded
        
        # Multi-task outputs
        outputs = {}
        for task, head in self.heads.items():
            outputs[task] = head(encoded)
        
        return outputs
    
    def enable_cache(self):
        """Enable KV cache for faster inference"""
        self.cache_enabled = True
        self.kv_cache = {}
    
    def clear_cache(self):
        """Clear KV cache"""
        self.kv_cache = {}
```

---

## 🤖 VLA에서의 활용

### 1. Instruction Grounding for VLA
```python
class InstructionGroundedVLA(nn.Module):
    """명령어 기반 VLA"""
    def __init__(self, vision_encoder, language_model):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        
        # Grounding network
        self.grounding = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Attention for grounding
        self.cross_attention = nn.MultiheadAttention(256, 8, batch_first=True)
        
        # Action generation
        self.action_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # 7-DOF action
        )
        
    def forward(self, image, instruction_ids):
        # Encode vision
        vision_features = self.vision_encoder(image)
        
        # Encode language
        language_outputs = self.language_model(instruction_ids)
        language_features = language_outputs['action_features']
        
        # Ground language in vision
        combined = torch.cat([vision_features, language_features], dim=-1)
        grounded = self.grounding(combined)
        
        # Cross-modal attention
        grounded = grounded.unsqueeze(1)
        attended, _ = self.cross_attention(
            grounded, grounded, grounded
        )
        
        # Generate action
        action = self.action_head(attended.squeeze(1))
        
        return action
    
    def interpret_instruction(self, instruction_text, tokenizer):
        """자연어 명령 해석"""
        # Parse instruction
        parsed = self.language_model.parse_instruction(instruction_text, tokenizer)
        
        # Extract key components
        interpretation = {
            'action': parsed['intent'],
            'object': parsed['semantic_roles'].get('object', []),
            'location': parsed['semantic_roles'].get('location', []),
            'modifiers': self.extract_modifiers(parsed)
        }
        
        return interpretation
    
    def extract_modifiers(self, parsed):
        """명령어에서 수식어 추출"""
        modifiers = []
        
        # Color modifiers
        colors = ['red', 'blue', 'green', 'yellow']
        for token in parsed['tokens']:
            if token.lower() in colors:
                modifiers.append(('color', token.lower()))
        
        # Size modifiers
        sizes = ['small', 'large', 'big', 'tiny']
        for token in parsed['tokens']:
            if token.lower() in sizes:
                modifiers.append(('size', token.lower()))
        
        return modifiers
```

### 2. Contextual Language Understanding
```python
class ContextualLanguageVLA(nn.Module):
    """문맥 인식 언어 처리 VLA"""
    def __init__(self, d_model=768):
        super().__init__()
        
        # Dialogue history encoder
        self.dialogue_encoder = nn.LSTM(
            d_model, d_model // 2,
            num_layers=2, bidirectional=True, batch_first=True
        )
        
        # Coreference resolution
        self.coref_resolver = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2)
        )
        
        # Ambiguity resolver
        self.ambiguity_scorer = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Context memory
        self.context_memory = []
        self.max_context_len = 10
        
    def forward(self, current_instruction, instruction_history=None):
        # Process current instruction
        current_features = self.encode_instruction(current_instruction)
        
        # Process dialogue history if available
        if instruction_history is not None:
            history_encoded, _ = self.dialogue_encoder(instruction_history)
            context = history_encoded[:, -1, :]
        else:
            context = torch.zeros_like(current_features)
        
        # Resolve coreferences
        combined = torch.cat([current_features, context], dim=-1)
        resolved = self.coref_resolver(combined)
        
        # Check for ambiguity
        ambiguity_score = self.ambiguity_scorer(resolved)
        
        # Update context memory
        self.update_context(current_features)
        
        return {
            'features': resolved,
            'ambiguity': ambiguity_score,
            'needs_clarification': ambiguity_score > 0.5
        }
    
    def encode_instruction(self, instruction):
        """Instruction encoding (placeholder)"""
        # This would use the actual language model
        return torch.randn(1, 768)
    
    def update_context(self, features):
        """Update context memory"""
        self.context_memory.append(features)
        if len(self.context_memory) > self.max_context_len:
            self.context_memory.pop(0)
    
    def resolve_pronouns(self, instruction, entities):
        """대명사 해결"""
        pronouns = ['it', 'that', 'this', 'them']
        resolved_instruction = instruction
        
        for pronoun in pronouns:
            if pronoun in instruction.lower():
                # Find most recent matching entity
                for entity in reversed(entities):
                    if self.is_compatible(pronoun, entity):
                        resolved_instruction = resolved_instruction.replace(
                            pronoun, entity['name']
                        )
                        break
        
        return resolved_instruction
    
    def is_compatible(self, pronoun, entity):
        """Check pronoun-entity compatibility"""
        if pronoun in ['it', 'that', 'this']:
            return entity['number'] == 'singular'
        elif pronoun == 'them':
            return entity['number'] == 'plural'
        return False
```

### 3. Multi-turn Dialogue for VLA
```python
class DialogueVLA(nn.Module):
    """대화형 VLA 시스템"""
    def __init__(self, language_model, vision_encoder):
        super().__init__()
        self.language_model = language_model
        self.vision_encoder = vision_encoder
        
        # Dialogue state tracking
        self.dialogue_state = {
            'history': [],
            'entities': [],
            'goals': [],
            'clarifications_needed': []
        }
        
        # Response generation
        self.response_generator = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 50000)  # Vocabulary size
        )
        
        # Clarification detection
        self.clarification_detector = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Need clarification: yes/no
        )
        
    def process_turn(self, user_input, visual_context):
        """Process single dialogue turn"""
        # Encode inputs
        language_features = self.language_model(user_input)
        vision_features = self.vision_encoder(visual_context)
        
        # Update dialogue state
        self.update_dialogue_state(user_input, language_features)
        
        # Check if clarification needed
        need_clarification = self.check_clarification(
            language_features, vision_features
        )
        
        if need_clarification:
            # Generate clarification question
            response = self.generate_clarification(language_features)
            action = None
        else:
            # Generate action
            action = self.generate_action(language_features, vision_features)
            response = self.generate_confirmation(action)
        
        # Update history
        self.dialogue_state['history'].append({
            'user': user_input,
            'robot': response,
            'action': action
        })
        
        return {
            'response': response,
            'action': action,
            'needs_clarification': need_clarification
        }
    
    def update_dialogue_state(self, user_input, features):
        """Update dialogue state"""
        # Extract entities
        entities = self.extract_entities(user_input)
        self.dialogue_state['entities'].extend(entities)
        
        # Extract goals
        goals = self.extract_goals(features)
        self.dialogue_state['goals'].extend(goals)
    
    def check_clarification(self, language_features, vision_features):
        """Check if clarification is needed"""
        combined = torch.cat([
            language_features['hidden_states'].mean(dim=1),
            vision_features.unsqueeze(0)
        ], dim=-1)
        
        logits = self.clarification_detector(combined)
        need_clarification = torch.argmax(logits, dim=-1).item() == 1
        
        return need_clarification
    
    def generate_clarification(self, language_features):
        """Generate clarification question"""
        # Simplified generation
        templates = [
            "Which {} do you mean?",
            "Do you want me to {} the {} one?",
            "Should I {} it to the {}?"
        ]
        
        # Select template based on context
        return templates[0].format("object")
    
    def generate_confirmation(self, action):
        """Generate action confirmation"""
        return f"I will perform action: {action}"
    
    def generate_action(self, language_features, vision_features):
        """Generate robot action"""
        # Simplified action generation
        return torch.randn(7)  # 7-DOF action
    
    def extract_entities(self, text):
        """Extract entities from text"""
        # Simplified entity extraction
        return []
    
    def extract_goals(self, features):
        """Extract goals from features"""
        # Simplified goal extraction
        return []
```

---

## 🔬 핵심 개념 정리

### 1. Tokenization Strategies
```python
class RobotTokenizer:
    """로봇 명령어 특화 토크나이저"""
    def __init__(self, vocab_file=None):
        # Basic vocabulary
        self.vocab = {
            '<PAD>': 0, '<UNK>': 1, '<CLS>': 2, '<SEP>': 3, '<MASK>': 4,
            # Action verbs
            'pick': 5, 'place': 6, 'move': 7, 'push': 8, 'pull': 9,
            'rotate': 10, 'open': 11, 'close': 12, 'grasp': 13, 'release': 14,
            # Objects
            'ball': 15, 'cube': 16, 'cylinder': 17, 'box': 18, 'bottle': 19,
            # Colors
            'red': 20, 'blue': 21, 'green': 22, 'yellow': 23,
            # Spatial relations
            'on': 24, 'under': 25, 'beside': 26, 'behind': 27, 'front': 28,
            'left': 29, 'right': 30, 'top': 31, 'bottom': 32,
            # Modifiers
            'slowly': 33, 'quickly': 34, 'carefully': 35, 'gently': 36
        }
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
    def tokenize(self, text):
        """Text to tokens"""
        tokens = text.lower().split()
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
    
    def decode(self, ids):
        """IDs to text"""
        tokens = [self.id_to_token.get(id, '<UNK>') for id in ids]
        return ' '.join(tokens)
    
    def add_special_tokens(self, token_ids, add_cls=True, add_sep=True):
        """Add special tokens"""
        if add_cls:
            token_ids = [self.vocab['<CLS>']] + token_ids
        if add_sep:
            token_ids = token_ids + [self.vocab['<SEP>']]
        return token_ids
```

### 2. Attention Visualization
```python
def visualize_attention(model, instruction, tokenizer):
    """Attention 패턴 시각화"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Tokenize
    tokens = tokenizer.tokenize(instruction)
    input_ids = torch.tensor([tokens])
    
    # Get attention weights
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        # Assuming model returns attention weights
        attention = outputs.get('attention_weights')
    
    if attention is not None:
        # Average over heads
        attention = attention.mean(dim=1)[0]  # [seq_len, seq_len]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attention.cpu().numpy(), 
                   xticklabels=tokenizer.decode(tokens).split(),
                   yticklabels=tokenizer.decode(tokens).split(),
                   cmap='Blues', ax=ax)
        ax.set_title('Attention Weights')
        plt.show()
```

---

## 🛠️ 실습 코드

### 완전한 로봇 명령어 처리 시스템
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

class RobotInstructionSystem:
    """완전한 로봇 명령어 처리 시스템"""
    def __init__(self, model_type='efficient'):
        # Initialize components
        self.tokenizer = RobotTokenizer()
        
        if model_type == 'efficient':
            self.language_model = EfficientRobotLM(
                vocab_size=len(self.tokenizer.vocab),
                d_model=256,
                n_layers=4
            )
        else:
            self.language_model = InstructionLM(
                base_model='bert',
                vocab_size=len(self.tokenizer.vocab)
            )
        
        # Instruction parser
        self.parser = InstructionParser()
        
        # Action mapper
        self.action_mapper = ActionMapper()
        
        # Safety checker
        self.safety_checker = SafetyChecker()
        
        # Dialogue manager
        self.dialogue_manager = DialogueManager()
        
    def process_instruction(self, instruction: str, context: Optional[Dict] = None):
        """Process single instruction"""
        # Tokenize
        tokens = self.tokenizer.tokenize(instruction)
        tokens = self.tokenizer.add_special_tokens(tokens)
        input_ids = torch.tensor([tokens])
        
        # Language model forward pass
        outputs = self.language_model(input_ids)
        
        # Parse instruction structure
        parsed = self.parser.parse(instruction, outputs)
        
        # Check safety
        is_safe, safety_msg = self.safety_checker.check(parsed)
        if not is_safe:
            return {
                'success': False,
                'message': safety_msg,
                'action': None
            }
        
        # Map to robot action
        action = self.action_mapper.map_to_action(parsed, context)
        
        # Generate response
        response = self.dialogue_manager.generate_response(parsed, action)
        
        return {
            'success': True,
            'parsed': parsed,
            'action': action,
            'response': response
        }
    
    def process_dialogue(self, conversation: List[str]):
        """Process multi-turn dialogue"""
        results = []
        context = {'history': [], 'entities': []}
        
        for turn in conversation:
            # Process with context
            result = self.process_instruction(turn, context)
            results.append(result)
            
            # Update context
            context['history'].append({
                'instruction': turn,
                'result': result
            })
            
            # Extract and update entities
            if result['success']:
                entities = self.extract_entities(result['parsed'])
                context['entities'].extend(entities)
        
        return results
    
    def extract_entities(self, parsed):
        """Extract entities from parsed instruction"""
        entities = []
        
        if 'object' in parsed:
            entities.append({
                'type': 'object',
                'value': parsed['object'],
                'properties': parsed.get('properties', {})
            })
        
        if 'location' in parsed:
            entities.append({
                'type': 'location',
                'value': parsed['location']
            })
        
        return entities

class InstructionParser:
    """명령어 구조 분석기"""
    def __init__(self):
        self.patterns = {
            'pick_and_place': r'pick (?:up )?the (\w+) (?:and )?place (?:it )?(?:on|in) the (\w+)',
            'move': r'move the (\w+) to (?:the )?(\w+)',
            'simple_action': r'(\w+) the (\w+)'
        }
    
    def parse(self, instruction, model_outputs):
        """Parse instruction into structured format"""
        import re
        
        parsed = {
            'raw': instruction,
            'intent': None,
            'object': None,
            'location': None,
            'properties': {}
        }
        
        # Pattern matching
        for pattern_name, pattern in self.patterns.items():
            match = re.match(pattern, instruction.lower())
            if match:
                if pattern_name == 'pick_and_place':
                    parsed['intent'] = 'pick_and_place'
                    parsed['object'] = match.group(1)
                    parsed['location'] = match.group(2)
                elif pattern_name == 'move':
                    parsed['intent'] = 'move'
                    parsed['object'] = match.group(1)
                    parsed['location'] = match.group(2)
                elif pattern_name == 'simple_action':
                    parsed['intent'] = match.group(1)
                    parsed['object'] = match.group(2)
                break
        
        # Extract properties (colors, sizes, etc.)
        parsed['properties'] = self.extract_properties(instruction)
        
        return parsed
    
    def extract_properties(self, instruction):
        """Extract object properties"""
        properties = {}
        
        # Colors
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white']
        for color in colors:
            if color in instruction.lower():
                properties['color'] = color
        
        # Sizes
        sizes = ['small', 'large', 'big', 'tiny', 'huge']
        for size in sizes:
            if size in instruction.lower():
                properties['size'] = size
        
        return properties

class ActionMapper:
    """파싱된 명령을 로봇 액션으로 매핑"""
    def __init__(self):
        self.action_templates = {
            'pick': {'gripper': 1.0, 'z': -0.1},
            'place': {'gripper': 0.0, 'z': 0.1},
            'move': {'x': 0.0, 'y': 0.0},
            'push': {'force': 5.0},
            'pull': {'force': -5.0}
        }
    
    def map_to_action(self, parsed, context=None):
        """Map parsed instruction to robot action"""
        action = torch.zeros(7)  # 7-DOF action
        
        intent = parsed.get('intent')
        if intent in self.action_templates:
            template = self.action_templates[intent]
            
            # Apply template
            if 'x' in template:
                action[0] = template['x']
            if 'y' in template:
                action[1] = template['y']
            if 'z' in template:
                action[2] = template['z']
            if 'gripper' in template:
                action[6] = template['gripper']
        
        # Adjust based on context
        if context and 'objects' in context:
            # Find target object position
            target_obj = parsed.get('object')
            for obj in context['objects']:
                if obj['name'] == target_obj:
                    action[:3] = torch.tensor(obj['position'])
        
        return action

class SafetyChecker:
    """안전성 검사"""
    def __init__(self):
        self.forbidden_actions = ['destroy', 'break', 'damage', 'throw']
        self.restricted_objects = ['human', 'person', 'animal']
    
    def check(self, parsed):
        """Check if instruction is safe"""
        # Check forbidden actions
        intent = parsed.get('intent', '')
        if any(forbidden in intent for forbidden in self.forbidden_actions):
            return False, f"Action '{intent}' is not allowed"
        
        # Check restricted objects
        obj = parsed.get('object', '')
        if any(restricted in obj for restricted in self.restricted_objects):
            return False, f"Cannot interact with '{obj}'"
        
        return True, "Safe"

class DialogueManager:
    """대화 관리"""
    def __init__(self):
        self.templates = {
            'confirmation': "I will {} the {}",
            'clarification': "Do you mean the {} {}?",
            'completion': "I have completed the task",
            'error': "I cannot {} because {}"
        }
    
    def generate_response(self, parsed, action):
        """Generate natural language response"""
        intent = parsed.get('intent', 'perform action')
        obj = parsed.get('object', 'object')
        
        # Generate appropriate response
        if action is not None:
            response = self.templates['confirmation'].format(intent, obj)
        else:
            response = self.templates['error'].format(intent, "action not recognized")
        
        return response

# Demo usage
def demo_instruction_system():
    # Initialize system
    system = RobotInstructionSystem(model_type='efficient')
    
    # Single instruction
    instruction = "Pick up the red ball and place it in the box"
    result = system.process_instruction(instruction)
    
    print(f"Instruction: {instruction}")
    print(f"Parsed: {result['parsed']}")
    print(f"Action: {result['action']}")
    print(f"Response: {result['response']}")
    
    # Multi-turn dialogue
    dialogue = [
        "I see a red ball and a blue cube",
        "Pick up the red one",
        "Now place it on the table",
        "Move the cube next to it"
    ]
    
    print("\n--- Multi-turn Dialogue ---")
    results = system.process_dialogue(dialogue)
    for i, (turn, result) in enumerate(zip(dialogue, results)):
        print(f"\nTurn {i+1}: {turn}")
        print(f"Response: {result['response']}")
        if result['action'] is not None:
            print(f"Action: {result['action'][:3]}")  # Show position only

if __name__ == "__main__":
    demo_instruction_system()
```

---

## 📈 다음 단계

### 1. 고급 언어 모델 기법
- **Prompt Engineering**: Few-shot learning
- **Instruction Tuning**: 명령어 수행 특화
- **Chain-of-Thought**: 추론 과정 명시

### 2. VLA 특화 개선
- **Embodied Language**: 신체화된 언어 이해
- **Spatial Language**: 공간 관계 표현
- **Temporal Grounding**: 시간적 순서 이해

### 3. 최신 연구
- **LLaMA-style Models**: 효율적 대규모 모델
- **Multimodal LLMs**: 비전-언어 통합 모델
- **Code Generation**: 로봇 프로그램 생성

---

## 💡 핵심 포인트

### ✅ 기억해야 할 것들
1. **GPT vs BERT**: 생성 vs 이해의 차이
2. **Attention 메커니즘**: 언어 이해의 핵심
3. **Context 관리**: 대화 연속성 유지
4. **Grounding**: 언어를 행동으로 연결

### ⚠️ 주의사항
1. **Ambiguity**: 모호한 명령 처리
2. **Safety**: 위험한 명령 필터링
3. **Efficiency**: 실시간 처리 요구

### 🎯 VLA 적용 시
1. **명확한 명령**: 구조화된 파싱
2. **Context awareness**: 상황 인식
3. **Feedback loop**: 명령 이해 확인

---

**다음 문서**: `08_imitation_learning.md` - 시연을 통한 학습