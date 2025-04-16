from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F
import time

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# User inputs
prompt = input("Enter your prompt: ")
temp_input = input("Enter temperature (0.1 - 2.0): ")

try:
    temperature = float(temp_input)
    if not (0.1 <= temperature <= 2.0):
        raise ValueError()
except ValueError:
    print("Invalid temperature. Using default 1.0")
    temperature = 1.0

top_k = 5
max_steps = 20

# Function: entropy
def calculate_entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-8)).item()

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

# Interactive token generation
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
print(f"\nStarting generation (max {max_steps} tokens):\n")

for step in range(max_steps):
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1).squeeze()

    entropy = calculate_entropy(probs)
    top_probs, top_indices = torch.topk(probs, top_k)

    print(f"\nStep {step + 1}")
    print("-" * 40)
    for i in range(top_k):
        token = tokenizer.decode(top_indices[i])
        prob = top_probs[i].item()
        bar = make_bar(prob)
        print(f"{i+1:>2}. '{token:<10}': {prob*100:5.2f}% |{bar}|")

    print(f"\nEntropy: {entropy:.3f} (lower = more confident)\n")

    choice = input("Pick token (1-5), press Enter for top choice, or type 'exit' to stop: ").strip().lower()
    if choice == 'exit':
        print("\n🚪 Exiting generation.")
        break
    elif choice in {'1','2','3','4','5'}:
        selected_index = int(choice) - 1
    else:
        selected_index = 0

    next_token_id = top_indices[selected_index].unsqueeze(0).unsqueeze(0)
    input_ids = torch.cat([input_ids, next_token_id], dim=1)

    # Print growing output
    output_so_far = tokenizer.decode(input_ids[0])
    print(f"\nGenerated so far: \033[1m{output_so_far}\033[0m")
    time.sleep(0.3)
