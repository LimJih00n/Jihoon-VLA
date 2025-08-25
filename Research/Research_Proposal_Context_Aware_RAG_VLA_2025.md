# Context-Aware RAG for Vision-Language-Action Models
## Research Proposal

---

## Research Question

**"What context should a robot retrieve, and when?"**

Current VLA models fail because they lack context. ELLMER (2025) showed RAG helps, but used naive retrieval. We propose **selective context management** for VLA.

---

## The Problem We Found

From our investigation:
- **ELLMER**: Uses RAG but retrieves everything (inefficient)
- **OpenVLA**: No context awareness (fails repeatedly)
- **Gap**: Nobody knows what context is actually useful for robots

### Key Insight from Our Analysis
```python
# Current RAG-VLA (ELLMER approach)
context = retrieve_all_relevant_docs()  # Too much, too slow

# Our approach: Selective Context
context = {
    'immediate': last_1s if is_reactive_task() else None,
    'task': last_10s if is_sequential_task() else None,  
    'knowledge': retrieve_only_if_uncertain()
}
```

---

## Core Innovation: Context-Aware Retrieval

Based on our research, we identified 3 critical context types:

| Context Type | When to Retrieve | What to Retrieve |
|-------------|------------------|------------------|
| **Immediate** (L1) | Always for manipulation | Last 1-3 actions |
| **Task** (L2) | For multi-step tasks | Subtask progress |
| **Knowledge** (L3) | When confidence < threshold | Similar past failures |

### The Key: **Adaptive Retrieval Policy**
```python
def should_retrieve(confidence, task_phase):
    if confidence < 0.7:  # Uncertain
        return 'knowledge'  # Retrieve past experiences
    elif task_phase == 'middle':
        return 'task'  # Check progress
    else:
        return 'immediate'  # Just recent context
```

---

## Why This Matters

Our folder research revealed:
1. **Retrieval latency kills real-time performance** (Critical_Analysis_2025.md)
2. **Most context is noise** - only 10-20% is useful
3. **Selective memory works** - failures are more informative than successes

---

## Specific Contribution

**"Context Selection Policy for RAG-VLA"**

Not just using RAG, but knowing:
- **WHEN** to retrieve (confidence-based triggering)
- **WHAT** to retrieve (failure-prioritized memory)
- **HOW MUCH** to retrieve (adaptive window sizing)

---

## Method (Based on Our Research)

1. **Baseline**: ELLMER approach (retrieve everything)
2. **Our Method**: Selective retrieval based on:
   - Action confidence scores
   - Task complexity estimation
   - Failure detection signals

3. **Metrics**:
   - Success rate improvement
   - Retrieval reduction (%)
   - Latency improvement (ms)

---

## Expected Results

From our analysis (RAG_VLA_Master_Strategy_2025.md):
```
Naive RAG: +10% success, +500ms latency
Selective RAG: +20% success, +50ms latency (10x faster)
```

---

## Feasibility

We already identified:
- **Base model**: OpenVLA (working)
- **RAG system**: ChromaDB (simple)
- **Key challenge**: Selection policy (our innovation)
- **Timeline**: 3-4 months

---

## One Sharp Focus

**We don't build a complex system.**
**We answer: "What context actually helps robots?"**

This is the missing piece between ELLMER (too much retrieval) and OpenVLA (no retrieval).

---

*Based on extensive research in this folder (15+ documents analyzed)*
*Focus: VLA + RAG + Context Management*