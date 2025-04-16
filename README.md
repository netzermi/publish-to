
## 🤖 Project Overview

This project is a **step-by-step token-level GPT-2 generation playground** where you can:

- Interactively select from top-k token options at each step  
- Visualize probabilities using colorful **ASCII bars**  
- Experiment with **temperature scaling** and observe its effect  
- Understand **entropy** as a measure of model confidence  
- Learn how language models make token decisions 🔍

---

## 📂 Files Included

### `interactive_gpt_generation.py`

🧠 Main script for **guided token-by-token text generation**.  
Let the model suggest words, inspect their probabilities, and decide yourself how the sentence continues.

#### Features:
- Top-k token selection (default = 5)
- Colored probability bars (green/yellow/red)
- Entropy indicator (how confident is GPT?)
- Option to exit any time with `'exit'`
- Beautifully readable output

---

### `temperature.py`

🔥 A demo focusing on how **temperature** affects GPT's output distribution.

#### Learn:
- How higher temperature → more randomness  
- How low temperature → more deterministic output  
- How logits become probabilities using softmax  
- How model creativity is mathematically controlled

Perfect for education or experimentation with sampling behavior.

---

## 🚀 How to Use

### Prerequisites

Install the required libraries:

```bash
pip install transformers torch
