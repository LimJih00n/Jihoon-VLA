# 🔗 VLA 연구 유용한 링크 모음
## 논문, 코드, 도구, 커뮤니티 총정리

---

## 📄 핵심 논문 및 프로젝트

### 🔥 Must-Know VLA Models
| 모델명 | 논문 링크 | 코드 링크 | 데모 | 설명 |
|--------|-----------|-----------|------|------|
| **OpenVLA** | [Paper](https://openvla.github.io/) | [GitHub](https://github.com/openvla/openvla) | [Demo](https://openvla.github.io/#demo) | 오픈소스 SOTA VLA |
| **RT-1** | [Paper](https://arxiv.org/abs/2212.06817) | [GitHub](https://github.com/google-research/robotics_transformer) | - | 구글의 첫 VLA |
| **RT-2** | [Paper](https://arxiv.org/abs/2307.15818) | [GitHub](https://github.com/google-deepmind/rt-2) | [Video](https://robotics-transformer-x.github.io/) | RT-1의 확장판 |
| **PaLM-E** | [Paper](https://arxiv.org/abs/2303.03378) | - | [Demo](https://palm-e.github.io/) | 구글의 멀티모달 |

### 🏗️ 기반 기술
| 기술 | 주요 논문 | 구현체 | 용도 |
|------|-----------|--------|------|
| **RAG** | [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) | [LangChain](https://github.com/langchain-ai/langchain) | 외부 지식 검색 |
| **CLIP** | [Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020) | [OpenAI CLIP](https://github.com/openai/CLIP) | Vision-Language |
| **Transformer** | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | [Transformers](https://github.com/huggingface/transformers) | 기본 아키텍처 |

---

## 💾 데이터셋

### 로봇 학습 데이터셋
| 이름 | 링크 | 크기 | 설명 |
|------|------|------|------|
| **RT-X** | [Dataset](https://robotics-transformer-x.github.io/) | 527K episodes | 다중 로봇 통합 데이터 |
| **DROID** | [GitHub](https://github.com/droid-dataset/droid) | 76K episodes | 표준화된 로봇 데이터 |
| **RH20T** | [Paper](https://arxiv.org/abs/2410.17194) | 20K+ hours | 2024 최신 데이터셋 |
| **BridgeData** | [Website](https://rail-berkeley.github.io/bridgedata/) | 65K demos | Berkeley 로봇 데이터 |

### 시뮬레이션 환경
| 환경 | 링크 | 특징 | 무료여부 |
|------|------|------|---------|
| **PyBullet** | [GitHub](https://github.com/bulletphysics/bullet3) | 빠른 물리 시뮬레이션 | ✅ 무료 |
| **MuJoCo** | [Website](https://mujoco.org/) | 정밀한 물리 엔진 | ✅ 무료 (2021년부터) |
| **Isaac Sim** | [NVIDIA](https://developer.nvidia.com/isaac-sim) | GPU 가속, 고품질 렌더링 | 💰 유료 |
| **RLBench** | [GitHub](https://github.com/stepjam/RLBench) | 100가지 로봇 태스크 | ✅ 무료 |

---

## 🛠️ 개발 도구

### 머신러닝 프레임워크
```python
ml_frameworks = {
    "PyTorch": {
        "링크": "https://pytorch.org/",
        "용도": "딥러닝 모델 구현",
        "VLA에서": "대부분의 VLA 모델이 PyTorch 기반"
    },
    
    "Transformers": {
        "링크": "https://huggingface.co/transformers/",
        "용도": "사전훈련된 모델 사용",
        "VLA에서": "CLIP, GPT 등 기반 모델 활용"
    },
    
    "LangChain": {
        "링크": "https://github.com/langchain-ai/langchain",
        "용도": "RAG 시스템 구축",
        "VLA에서": "Context-Aware RAG 구현"
    }
}
```

### 실험 관리
| 도구 | 링크 | 용도 | 가격 |
|------|------|------|------|
| **Weights & Biases** | [wandb.ai](https://wandb.ai/) | 실험 추적, 시각화 | 무료 (개인) |
| **MLflow** | [mlflow.org](https://mlflow.org/) | 모델 라이프사이클 관리 | 무료 |
| **TensorBoard** | [tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard) | 기본 시각화 | 무료 |

### 벡터 데이터베이스 (RAG용)
| 이름 | 링크 | 특징 | 가격 |
|------|------|------|------|
| **ChromaDB** | [chromadb.com](https://www.trychroma.com/) | 간단한 설정, 로컬 실행 | 무료 |
| **Qdrant** | [qdrant.tech](https://qdrant.tech/) | 고성능, 스케일링 | 무료 + 유료 |
| **Pinecone** | [pinecone.io](https://www.pinecone.io/) | 관리형 서비스 | 유료 |
| **Weaviate** | [weaviate.io](https://weaviate.io/) | 오픈소스, 완전 기능 | 무료 + 유료 |

---

## 📚 학습 리소스

### 📖 온라인 강의
| 강의명 | 제공기관 | 링크 | 레벨 | 시간 |
|--------|----------|------|------|------|
| **CS231N** | Stanford | [YouTube](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) | 중급 | ~20시간 |
| **CS224N** | Stanford | [YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ) | 중급 | ~25시간 |
| **Deep RL** | Berkeley | [CS285](http://rail.eecs.berkeley.edu/deeprlcourse/) | 고급 | ~30시간 |
| **Robot Learning** | CMU | [16-831](https://www.cs.cmu.edu/~./16831-f14/) | 고급 | ~25시간 |

### 📚 책
```python
essential_books = {
    "입문서": [
        {
            "제목": "Hands-On Machine Learning",
            "저자": "Aurélien Géron",  
            "링크": "https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/",
            "추천이유": "실습 중심, 코드 예제 풍부"
        }
    ],
    
    "이론서": [
        {
            "제목": "Deep Learning", 
            "저자": "Ian Goodfellow",
            "링크": "https://www.deeplearningbook.org/",
            "추천이유": "딥러닝 이론의 바이블"
        }
    ],
    
    "로보틱스": [
        {
            "제목": "Robotics: Modelling, Planning and Control",
            "저자": "Bruno Siciliano",
            "추천이유": "로봇 공학 전반적 이해"
        }
    ]
}
```

### 🎥 YouTube 채널
| 채널명 | 링크 | 특징 | 업데이트 빈도 |
|--------|------|------|---------------|
| **Yannic Kilcher** | [Channel](https://www.youtube.com/@YannicKilcher) | 최신 논문 리뷰 | 주 2-3회 |
| **Two Minute Papers** | [Channel](https://www.youtube.com/@TwoMinutePapers) | 논문을 쉽게 설명 | 주 2회 |
| **3Blue1Brown** | [Channel](https://www.youtube.com/@3blue1brown) | 수학적 직관 | 불규칙 |
| **Lex Fridman** | [Channel](https://www.youtube.com/@lexfridman) | AI 연구자 인터뷰 | 주 1회 |

---

## 🌐 커뮤니티 및 포럼

### 연구 커뮤니티
| 플랫폼 | 링크 | 특징 | 참여 방법 |
|--------|------|------|-----------|
| **Papers with Code** | [paperswithcode.com](https://paperswithcode.com/) | 논문 + 코드 함께 | 논문/코드 업로드 |
| **Hugging Face** | [huggingface.co](https://huggingface.co/) | 모델 공유 플랫폼 | 모델/데이터셋 업로드 |
| **Reddit r/MachineLearning** | [r/MachineLearning](https://www.reddit.com/r/MachineLearning/) | ML 뉴스, 토론 | 포스트, 댓글 |
| **AI Twitter** | Twitter | 연구자들의 실시간 토론 | 팔로우, 리트윗 |

### Discord/Slack 그룹
```python
community_groups = {
    "Hugging Face Discord": {
        "링크": "https://hf.co/join/discord",
        "특징": "모델 개발 토론",
        "활성도": "매우 높음"
    },
    
    "EleutherAI Discord": {
        "링크": "https://discord.gg/zBGx3azzUn", 
        "특징": "오픈소스 AI 연구",
        "활성도": "높음"
    },
    
    "OpenAI Discord": {
        "특징": "OpenAI API 관련",
        "활성도": "높음"
    }
}
```

### 학회 및 워크샵
| 학회 | 날짜(일반적) | 웹사이트 | VLA 관련성 |
|------|--------------|----------|------------|
| **NeurIPS** | 12월 | [neurips.cc](https://neurips.cc/) | ⭐⭐⭐⭐⭐ |
| **ICML** | 7월 | [icml.cc](https://icml.cc/) | ⭐⭐⭐⭐⭐ |
| **ICLR** | 4-5월 | [iclr.cc](https://iclr.cc/) | ⭐⭐⭐⭐⭐ |
| **CoRL** | 10-11월 | [robot-learning.org](https://www.robot-learning.org/) | ⭐⭐⭐⭐⭐ |
| **RSS** | 7월 | [roboticsconference.org](https://roboticsconference.org/) | ⭐⭐⭐⭐ |
| **ICRA** | 5월 | [icra.org](https://www.icra.org/) | ⭐⭐⭐⭐ |
| **IROS** | 9-10월 | [iros.org](https://www.iros.org/) | ⭐⭐⭐ |

---

## 🔬 연구 도구

### 논문 관리
| 도구 | 링크 | 특징 | 가격 |
|------|------|------|------|
| **Zotero** | [zotero.org](https://www.zotero.org/) | 무료, 강력한 기능 | 무료 |
| **Mendeley** | [mendeley.com](https://www.mendeley.com/) | 소셜 기능 | 무료 + 유료 |
| **Notion** | [notion.so](https://www.notion.so/) | 통합 워크스페이스 | 무료 + 유료 |
| **Obsidian** | [obsidian.md](https://obsidian.md/) | 그래프 뷰, 연결성 | 무료 + 유료 |

### 논문 검색
```python
search_engines = {
    "arXiv.org": {
        "링크": "https://arxiv.org/",
        "특징": "최신 preprint",
        "검색팁": "카테고리 cs.RO (Robotics), cs.AI"
    },
    
    "Google Scholar": {
        "링크": "https://scholar.google.com/",
        "특징": "인용수 기반 랭킹",
        "검색팁": "인용수 높은 논문부터"
    },
    
    "Semantic Scholar": {
        "링크": "https://www.semanticscholar.org/",
        "특징": "AI 기반 추천",
        "검색팁": "관련 논문 자동 추천"
    },
    
    "Connected Papers": {
        "링크": "https://www.connectedpapers.com/",
        "특징": "논문 간 관계 시각화",
        "검색팁": "핵심 논문 하나로 시작"
    }
}
```

### 수식 및 그림 도구
| 도구 | 링크 | 용도 | 학습곡선 |
|------|------|------|----------|
| **LaTeX** | [overleaf.com](https://www.overleaf.com/) | 논문 작성, 수식 | 보통 |
| **Draw.io** | [app.diagrams.net](https://app.diagrams.net/) | 다이어그램, 플로우차트 | 쉬움 |
| **TikZ** | [tikz.net](https://tikz.net/) | 고품질 그래프/다이어그램 | 어려움 |
| **Matplotlib** | [matplotlib.org](https://matplotlib.org/) | 파이썬 시각화 | 보통 |
| **Plotly** | [plotly.com](https://plotly.com/) | 인터랙티브 그래프 | 보통 |

---

## 💻 개발 환경

### GPU 클라우드 서비스
| 서비스 | 링크 | GPU 종류 | 가격(시간당) | 특징 |
|--------|------|----------|-------------|------|
| **Google Colab** | [colab.research.google.com](https://colab.research.google.com/) | T4, V100 | 무료 + $10/월 | 쉬운 시작 |
| **Kaggle** | [kaggle.com](https://www.kaggle.com/) | P100, T4 | 무료 (주 30시간) | 커널 환경 |
| **Paperspace** | [paperspace.com](https://www.paperspace.com/) | RTX 4000+ | $0.4+/시간 | Jupyter 환경 |
| **AWS** | [aws.amazon.com](https://aws.amazon.com/) | A100, V100 | $1+/시간 | 완전한 제어 |
| **Lambda Labs** | [lambdalabs.com](https://lambdalabs.com/) | A100, H100 | $1.1+/시간 | ML 특화 |

### 개발 도구
```bash
# 필수 Python 패키지들
essential_packages = [
    "torch",              # PyTorch
    "transformers",       # Hugging Face
    "langchain",          # RAG framework
    "chromadb",           # Vector database
    "matplotlib",         # 기본 시각화
    "plotly",            # 인터랙티브 그래프  
    "jupyter",           # 노트북 환경
    "wandb",             # 실험 관리
    "opencv-python",     # 컴퓨터 비전
    "numpy",             # 수치 계산
    "pandas",            # 데이터 처리
    "scikit-learn",      # 머신러닝
    "pybullet",          # 로봇 시뮬레이션
    "gym"                # RL 환경
]
```

---

## 📊 유용한 웹사이트

### 논문 트렌드 분석
| 사이트 | 링크 | 기능 |
|--------|------|------|
| **Papers with Code Trends** | [paperswithcode.com/trends](https://paperswithcode.com/trends) | 분야별 논문 동향 |
| **arXiv Sanity** | [arxiv-sanity-lite.com](http://arxiv-sanity-lite.com/) | arXiv 논문 추천 |
| **AI Research Navigator** | [ai.googleblog.com](https://ai.googleblog.com/) | 구글 AI 블로그 |

### 벤치마크 및 리더보드
| 벤치마크 | 링크 | 분야 |
|----------|------|------|
| **RLBench** | [rlbench.github.io](https://sites.google.com/view/rlbench) | 로봇 조작 태스크 |
| **CALVIN** | [calvin-benchmark.github.io](https://calvin-benchmark.github.io/) | Long-horizon 태스크 |
| **Meta-World** | [meta-world.github.io](https://meta-world.github.io/) | 다양한 조작 태스크 |

---

## 🏢 연구기관 및 랩

### 해외 주요 연구기관
| 기관 | 주요 연구자 | 웹사이트 | 주요 연구 |
|------|-------------|----------|-----------|
| **Google DeepMind** | Sergey Levine | [deepmind.com](https://deepmind.com/) | RT-1, RT-2, Gato |
| **OpenAI** | - | [openai.com](https://openai.com/) | GPT, DALL-E, 로봇 연구 |
| **UC Berkeley** | Pieter Abbeel | [bair.berkeley.edu](https://bair.berkeley.edu/) | BAIR, 로봇 학습 |
| **Stanford** | Fei-Fei Li | [ai.stanford.edu](https://ai.stanford.edu/) | HAI, 컴퓨터 비전 |
| **MIT CSAIL** | - | [csail.mit.edu](https://csail.mit.edu/) | 로보틱스, AI |
| **CMU RI** | - | [ri.cmu.edu](https://www.ri.cmu.edu/) | 로보틱스 인스티튜트 |

### 국내 연구기관
| 기관 | 웹사이트 | 특징 |
|------|----------|------|
| **NAVER LABS** | [naverlabs.com](https://www.naverlabs.com/) | 자율주행, 로봇 |
| **삼성 리서치** | [research.samsung.com](https://research.samsung.com/) | AI, 로보틱스 |
| **카카오브레인** | [kakaobrain.com](https://kakaobrain.com/) | AI 연구 |
| **KAIST AI** | [ai.kaist.ac.kr](https://ai.kaist.ac.kr/) | AI 대학원 |
| **포항공대 AI** | [ai.postech.ac.kr](https://ai.postech.ac.kr/) | AI 대학원 |

---

## 🚀 실용적 팁

### 효율적인 정보 수집
```python
productivity_tips = {
    "RSS_구독": [
        "Google AI Blog",
        "OpenAI Blog", 
        "DeepMind Blog",
        "Papers with Code"
    ],
    
    "Twitter_팔로우": [
        "@GoogleAI",
        "@OpenAI", 
        "@DeepMind",
        "@paperswithcode"
    ],
    
    "주간_체크": [
        "arXiv cs.RO 새 논문",
        "Papers with Code 트렌딩",
        "Reddit r/MachineLearning 주요 글"
    ]
}
```

### 북마크 추천 구조
```
📂 VLA Research/
├── 📂 Papers/
│   ├── arXiv.org
│   ├── Google Scholar  
│   └── Connected Papers
├── 📂 Code/
│   ├── OpenVLA GitHub
│   ├── Hugging Face
│   └── Papers with Code
├── 📂 Tools/
│   ├── Weights & Biases
│   ├── Zotero
│   └── Overleaf
├── 📂 Communities/
│   ├── Reddit ML
│   ├── Hugging Face Discord
│   └── AI Twitter Lists
└── 📂 Learning/
    ├── Stanford CS courses
    ├── YouTube channels
    └── Online books
```

---

## 📱 모바일 앱

### 논문 읽기
| 앱 | 플랫폼 | 특징 |
|-----|--------|------|
| **Papers** | iOS | PDF 주석, 동기화 |
| **Zotero** | iOS/Android | 논문 관리 |
| **Mendeley** | iOS/Android | 소셜 기능 |

### 학습 관리  
| 앱 | 플랫폼 | 특징 |
|-----|--------|------|
| **Notion** | iOS/Android | 통합 워크스페이스 |
| **Obsidian** | iOS/Android | 그래프 뷰 |
| **Forest** | iOS/Android | 집중 시간 관리 |

---

## 🔄 정기적으로 체크할 것들

### 매일
- [ ] arXiv cs.RO, cs.AI 새 논문
- [ ] Hugging Face 일일 논문
- [ ] Twitter AI 트렌드

### 매주  
- [ ] Papers with Code 트렌딩
- [ ] Reddit r/MachineLearning 주간 하이라이트
- [ ] YouTube 채널들 새 영상

### 매월
- [ ] 주요 학회 발표 자료
- [ ] 새로운 데이터셋 출시
- [ ] 툴/프레임워크 업데이트

---

**이 리스트는 계속 업데이트됩니다!**

새로운 유용한 링크를 발견하면 언제든지 추가해주세요! 🚀

---

*Last Updated: 2025-08-24*  
*Maintained by: VLA Research Community*