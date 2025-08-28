# peradrm2.py
uniwersalny moduł peradrm2.py do gier planszowych z DRM 3.0
Here's a concise `README.md` in English for the `peradrm2.py` module, as requested.

### PERA-DRM 2.0: Dynamic Rules Matrix for Board Games 🧠🎮

PERA-DRM 2.0 is a versatile Python module designed to enhance board game AI by integrating a **Permutation-Equivariant Neural Network (PERA-Net)** with an advanced **Dynamic Rules Matrix (DRM 3.0)** system. This combination allows for a powerful, adaptable, and robust AI capable of learning and evolving strategies in real-time.

-----

### Features ✨

  * **PERA-Net**: A specialized neural network architecture for board games, including residual and attention blocks. It's **permutation-equivariant**, meaning it inherently understands and respects board symmetries (like rotations and flips), which significantly improves learning efficiency.
  * **DRM 3.0**: An advanced system for dynamically generating and managing "rules" or strategic heuristics. It features:
      * **Adaptive Strength**: Rules' influence and strength change based on their effectiveness and external rewards.
      * **Anti-Stagnation & Revival**: Mechanisms to detect strategic stagnation and introduce new, mutated rules to ensure continuous evolution and avoid getting stuck in local optima.
      * **Curiosity Bonus**: Rewards less-used rules to encourage exploration of new strategies.
  * **Enhanced PUCT**: A custom **Polynomial Upper Confidence Trees** algorithm that incorporates a **DRM bonus**. This bonus dynamically biases the tree search towards moves that align with the current effective rules, blending the network's deep search with the DRM's evolving strategic insights.

-----

### Core Concepts 📖

  * **PERA-Net**: Uses an equivariant convolutional neural network to process game states and output policy (move probabilities) and value (game outcome prediction). This architecture is efficient for games with grid-like structures, like Go or Chess.
  * **DRM System**: Operates in cycles. In each cycle, it updates the strength of its rules, potentially mutates existing ones, and adds new ones. The system continuously adapts to find a set of rules that lead to success.
  * **Integration**: The DRM and PERA-Net work in tandem. The network provides a broad, deep understanding of the game state, while the DRM provides tactical, evolving guidance. The PUCT algorithm combines these two sources of information to make intelligent move decisions.

-----

### Getting Started 🚀

This module is a complete, self-contained system. You can test it directly by running the `peradrm2.py` file. The `if __name__ == "__main__":` block includes tests for each component and the full integration, demonstrating the system's core functionality.

**Requirements:**

  * `torch`
  * `numpy`

**Example:**

```bash
python peradrm2.py
```

-----

### Module Structure 📁

The code is organized into logical sections:

1.  **DRM 3.0 Component**: `Rule` and `DRM3System` classes. Handles dynamic rule management, adaptation, and anti-stagnation mechanisms.
2.  **PERA-Net Component**: `EquivariantConv2d`, `ResidualBlock`, `AttentionBlock`, and `PERANet` classes. Defines the neural network architecture.
3.  **Integration**: `Node` and `EnhancedPUCT` classes. Contains the logic for the tree search algorithm, which uses both the network and the DRM system.
4.  **Testing and Utilities**: A `main` block (`if __name__ == "__main__":`) that demonstrates how to initialize and run the system.


