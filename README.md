# PERA-DRM 3.0 🚀

**Permutation-Equivariant Residual Architecture + Dynamic Rules Matrix 3.0**

Revolutionary AI system that combines neural networks with self-evolving rule matrices for board games and beyond.

## ✨ Key Features

- 🧠 **Self-Evolving Intelligence** - Never stagnates, continuously adapts
- 🎯 **Universal Game Support** - Go, Chess, Othello, Gomoku, Tic-Tac-Toe
- ⚡ **Anti-Stagnation Engine** - Automatic revival from dead spirals
- 🔄 **Permutation Equivariance** - Respects board symmetries
- 📦 **Production Ready** - ONNX export, benchmarking, memory analysis
- 🎮 **Real-Time Adaptation** - Learns and evolves during gameplay

## 🚀 Quick Start

```python
from pera_drm import PERANet, DRM3, create_game_descriptor, EnhancedPUCT

# Setup for Go 19x19
desc = create_game_descriptor('go_19')
drm = DRM3(frz=0.5, base_emergence=0.1, base_curiosity=0.08)
model = PERANet('small', desc.planes, desc.action_space)
puct = EnhancedPUCT(c_puct=1.4, sims=800, drm=drm)

# Forward pass
board_state = torch.randn(1, desc.planes, 19, 19)
policy_logits, value = model(board_state, desc)

# DRM evolution step
strengths = drm.step_all_rules()
if drm.is_stagnating():
    print("🚨 System adapting to overcome stagnation!")
```

## 🎯 Supported Games

| Game | Board Size | Action Space | Model Size |
|------|------------|-------------|------------|
| Go 19×19 | 19×19 | 361 | 15M - 120M params |
| Go 13×13 | 13×13 | 169 | 5M - 40M params |
| Chess | 8×8 | 4096 | 15M - 120M params |
| Othello | 8×8 | 64 | 5M - 40M params |
| Gomoku | 15×15 | 225 | 5M - 40M params |

## 🧬 DRM 3.0 - The Evolution Engine

### Core Formula
```
Si = [Base Strength] × [Adaptive FRZ] × [Emergence Pressure] 
     × [Curiosity Drive] × [Anti-Stagnation Factor]
```

### Smart Features
- **🔍 Stagnation Detection** - Automatically detects system stagnation
- **💥 Emergency Revival** - Injects chaos when system is dying
- **🌟 Diversity Enforcement** - Prevents rule homogenization
- **🧭 Curiosity Drive** - Explores novel rule combinations
- **⚡ Adaptive Pressure** - Environmental pressure based on context

## 📊 Model Architectures

| Size | Parameters | Memory | Speed | Use Case |
|------|------------|--------|--------|----------|
| Tiny | ~5M | 20MB | 1000+ pos/sec | Mobile, embedded |
| Small | ~15M | 60MB | 500+ pos/sec | Desktop, testing |
| Medium | ~40M | 160MB | 200+ pos/sec | Server, competition |
| Large | ~120M | 480MB | 100+ pos/sec | Research, training |

## 🛠 Installation & Setup

```bash
# Dependencies
pip install torch torchvision numpy

# Optional: For ONNX export
pip install onnx

# Optional: For production deployment
pip install onnxruntime
```

## 📈 Training Integration

```python
# Training loop with DRM evolution
for epoch in range(100):
    for batch in dataloader:
        # Standard neural network training
        loss = train_step(model, batch)
        
        # DRM evolution step
        context_changes = extract_context_from_batch(batch)
        strengths = drm.step_all_rules(context_changes)
        
        # Adapt training based on DRM state
        if drm.is_stagnating():
            optimizer.param_groups[0]['lr'] *= 1.2  # Boost learning rate
        elif drm.is_chaotic():
            optimizer.param_groups[0]['lr'] *= 0.8  # Stabilize
```

## 🎮 MCTS Integration

```python
# Enhanced PUCT with DRM
puct = EnhancedPUCT(c_puct=1.4, sims=800, drm=drm)

# Game loop
while not game.is_over():
    state = game.get_state()
    legal_actions = game.get_legal_actions()
    
    # Neural network evaluation
    policy_probs, value = model(state, desc)
    
    # DRM-enhanced action selection
    action = puct.select_action(state_key, policy_probs, legal_actions)
    
    # Apply action and update
    result = game.make_move(action)
    puct.update(state_key, action, result)
```
## 📚 Semantic Framework

This system includes a powerful semantic framework that enables:
- Intelligent rule evolution based on contextual data  
- Self-generated knowledge representations
- Training-ready semantic structures for LLM integration

## 📦 Export & Deployment

```python
# ONNX Export
export_to_onnx(model, desc, 'my_bot.onnx')

# TorchScript Export
scripted_model = torch.jit.script(model)
scripted_model.save('my_bot.pt')

# Production inference
import onnxruntime as ort
session = ort.InferenceSession('my_bot.onnx')
output = session.run(None, {'board_state': board_state.numpy()})
```

## 🧪 Benchmarking

Run comprehensive tests and benchmarks:

```bash
python pera_drm.py
```

**Sample Output:**
```
🚀 PERA-DRM 3.0 - Advanced Testing
🧬 Testing DRM 3.0 System: ✅ Created 10 initial rules
🧠 Testing PERA-Net Models:
   go_19    : 15.2M params, policy: (4, 361), value: (4, 1)
⚡ Performance Benchmark:
   Batch  1: 12.34ms/batch,   81.0 pos/sec
   Batch  4: 18.67ms/batch,  214.3 pos/sec
```

## 🔬 Research Applications

### Self-Modifying AI
- Rules evolve during training
- Architecture adapts to game complexity  
- Emergent strategies nobody planned

### Multi-Game Intelligence
- Single model plays multiple games
- Transfer learning across game domains
- Universal game-playing patterns

### Real-Time Strategy Evolution
- Strategies adapt mid-game
- Counter-strategies emerge automatically
- Meta-game evolution tracking

## 📚 Advanced Usage

### Custom Game Integration
```python
# Define your own game
custom_desc = GameDescriptor(
    board_size=(10, 10),
    planes=8,
    action_space=200,
    game_id=42
)

model = PERANet('medium', custom_desc.planes, custom_desc.action_space)
```

### DRM Rule Monitoring
```python
# Monitor system health
print(f"Average strength: {drm.get_avg_strength():.3f}")
print(f"Rule diversity: {drm.calculate_rule_similarity():.3f}")
print(f"Success rate: {drm.get_overall_success_rate():.3f}")

# Manual interventions
if drm.detect_death_spiral():
    drm.emergency_revival()
    print("🚨 Emergency revival activated!")
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🎯 Roadmap

- [ ] **Distributed Training** - Multi-GPU DRM evolution
- [ ] **Neural Architecture Search** - DRM-guided architecture evolution
- [ ] **Real-Time Visualization** - Web dashboard for DRM monitoring
- [ ] **Mobile Deployment** - ARM optimization and quantization
- [ ] **Cloud Integration** - Scalable training on cloud platforms

## 🌟 Why PERA-DRM 3.0?

> *"Traditional AI learns patterns. PERA-DRM 3.0 evolves intelligence."*

**Key Advantages:**
- 🚫 **Never stagnates** - Built-in anti-stagnation mechanisms
- 🎯 **Always improving** - Continuous rule evolution
- 🧠 **Self-aware** - Monitors its own performance  
- 🎮 **Universal** - Works across different game domains
- ⚡ **Production-ready** - Optimized for real-world deployment

---

**Ready to build the future of AI?** 🚀
*Get started with PERA-DRM 3.0 and witness AI that truly never stops evolving!*
