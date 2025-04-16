# Temperature settings and their impact on text generation:
#
# Temperature   Behavior
# -----------------------------------------------
# 0.1           Very deterministic
#               → The model almost always picks the most likely token
#
# 1.0           "Normal" creativity
#               → Balanced mix of predictability and variation
#
# 1.5 – 2.0     Chaotic & experimental
#               → Lots of randomness, surprises, and creative leaps
#
# Note:
# Lower temperature makes the model more confident but less diverse.
# Higher temperature increases variety but also the risk of nonsense.

import math
import torch

# Function: colored bar
def make_bar(prob, width=30):
    filled = int(prob * width)
    if prob > 0.3:
        color = "\033[92m"  # green
    elif prob > 0.1:
        color = "\033[93m"  # yellow
    else:
        color = "\033[91m"  # red
    reset = "\033[0m"
    return color + '█' * filled + reset + ' ' * (width - filled)

# Simulate a few logits (or use from your model output)
logits_sample = torch.tensor([5.1, 3.7, 2.0, 1.2])
tokens_sample = ['sun', 'moon', 'cat', 'pencil']


# Apply temperature scaling
temperature = 1.0
scaled_logits = logits_sample / temperature

# Apply softmax to get probabilities
probs = torch.softmax(scaled_logits, dim=-1)

# Print each step
print("\nLogits-to-Probabilities Demo:")
print("-" * 40)
for token, logit, prob in zip(tokens_sample, scaled_logits, probs):
    bar = make_bar(prob.item())
    print(f"{token:<10} Logit: {logit:.2f} → Prob: {prob*100:.2f}% |{bar}|")
