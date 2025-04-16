from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# 🔁 Get user input
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

# Get logits for the next token (after the prompt)
next_token_logits = logits[0, -1] / temperature

# Get probabilities
probs = F.softmax(next_token_logits, dim=-1)

# Get top-k predictions
top_k = 5
top_probs, top_indices = torch.topk(probs, top_k)

# 🔡 Decode and display
print(f"\nPrompt: {prompt}")
print(f"Temperature: {temperature}")
print("\nTop Word Predictions:\n")

for i in range(top_k):
    token_str = tokenizer.decode(top_indices[i])
    prob = top_probs[i].item() * 100
    print(f"  {i+1}. '{token_str}':\t{prob:.2f}%")
