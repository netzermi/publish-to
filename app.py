from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Get user input
prompt = input("Enter your prompt: ")
temp_input = input("Enter temperature (0.1 - 2.0): ")

try:
    temperature = float(temp_input)
    if not (0.1 <= temperature <= 2.0):
        raise ValueError()
except ValueError:
    print("Invalid temperature. Using default 1.0")
    temperature = 1.0

# Tokenize prompt
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]

# Get model output
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get logits for the next token
next_token_logits = logits[0, -1] / temperature

# Compute probabilities
probs = F.softmax(next_token_logits, dim=-1)

# Get top-k tokens
top_k = 5
top_probs, top_indices = torch.topk(probs, top_k)

# Color codes
def color_bar(p):
    if p > 0.3:
        color = "\033[92m"  # green
    elif p > 0.1:
        color = "\033[93m"  # yellow
    else:
        color = "\033[91m"  # red
    reset = "\033[0m"
    return color, reset

# Create colored bar visualization
def make_bar(p, bar_width=30):
    filled = int(p * bar_width)
    color, reset = color_bar(p)
    bar = color + '█' * filled + reset + ' ' * (bar_width - filled)
    return bar

# Display output
print(f"\nPrompt: {prompt}")
print(f"Temperature: {temperature:.2f}")
print("\nTop Word Predictions:\n")

for i in range(top_k):
    token_str = tokenizer.decode(top_indices[i]).replace('\n', '\\n').replace('\t', '\\t')
    prob = top_probs[i].item()
    bar = make_bar(prob)
    print(f"{i+1:>2}. '{token_str:<10}': {prob*100:5.2f}% |{bar}|")
