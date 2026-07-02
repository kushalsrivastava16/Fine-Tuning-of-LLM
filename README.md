# Mistral-7B LeetCode Fine-Tuning (QLoRA)

Fine-tuning **Mistral-7B-Instruct-v0.1** on a LeetCode problems dataset using **QLoRA** (4-bit quantization + LoRA) to produce a code-generation assistant capable of solving algorithmic programming problems.

## Overview

This project fine-tunes a 7-billion-parameter LLM on ~21,000 LeetCode problem-solution pairs. By combining 4-bit quantization (BitsAndBytes) with Low-Rank Adaptation (LoRA via PEFT), the full model is trainable on a single A100 GPU while only updating 0.047% of parameters (3.4M out of 7.2B).

## Results

| Metric | Value |
|---|---|
| Eval Loss (3 epochs) | 2.379 |
| Trainable Parameters | 3,407,872 (0.047%) |
| Total Parameters | 7,245,139,968 |
| Avg Generation Speed | ~2.3 tokens/second |
| GPU | NVIDIA A100-SXM4-40GB |

### Sample Output

**Prompt:** `Write a Python function to check if a number is prime.`

```python
def is_prime(num):
    if num < 2:
        return False
    for i in range(2, num):
        if num % i == 0:
            return False
    return True
```

## Dataset

The dataset (`data_leetcode_problems.csv`) contains LeetCode problem entries in instruction-tuning format:

| Column | Description |
|---|---|
| `instruction` | Problem statement |
| `input` | Optional context/constraints |
| `output` | Reference solution |
| `text` | Combined prompt-response string used for training |

- Total examples: ~20,971
- Train / Eval split: 85% / 15%
- Max token length: 360

## Architecture & Training Details

### Model
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.1`
- **Quantization:** 4-bit (NF4) with double quantization, compute dtype float16

### LoRA Configuration
| Parameter | Value |
|---|---|
| Rank (`r`) | 16 |
| Alpha (`lora_alpha`) | 32 |
| Dropout | 0.1 |
| Target modules | `q_proj`, `v_proj` |
| Bias | none |
| Task type | CAUSAL_LM |

### Training Arguments
| Parameter | Value |
|---|---|
| Epochs | 3 |
| Batch size (per device) | 8 |
| Gradient accumulation steps | 4 (effective batch = 32) |
| Learning rate | 2e-5 |
| LR scheduler | Cosine |
| Warmup steps | 500 |
| Precision | bf16 |
| Optimizer | `adamw_torch_fused` |
| Gradient checkpointing | Enabled |

### Evaluation Metrics
- **BLEU** — token-level n-gram precision
- **ROUGE-L** — longest common subsequence recall
- **METEOR** — semantic similarity via stemming and synonym matching

## Project Structure

```
.
├── majorproject.ipynb       # Main training and evaluation notebook
├── data_leetcode_problems.csv  # LeetCode dataset (instruction-tuning format)
└── fine_tuned_mistral/      # Saved model weights and tokenizer (post-training)
```

## Requirements

```
torch
transformers
peft
datasets
accelerate
bitsandbytes
evaluate
rouge_score
nltk
scikit-learn
pandas
```

Install all dependencies:

```bash
pip install torch transformers peft datasets accelerate bitsandbytes evaluate rouge_score nltk scikit-learn pandas
```

## Usage

### 1. Setup (Google Colab recommended — requires A100 or comparable GPU)

```python
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")  # Never commit real tokens
```

### 2. Train

Open `majorproject.ipynb` and run all cells sequentially. The notebook will:
1. Load and quantize `Mistral-7B-Instruct-v0.1`
2. Tokenize the LeetCode dataset
3. Apply LoRA adapters via PEFT
4. Fine-tune for 3 epochs with the Hugging Face `Trainer`
5. Save the adapted model to `./fine_tuned_mistral`

### 3. Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_mistral")
model = AutoModelForCausalLM.from_pretrained(
    "./fine_tuned_mistral",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

prompt = "### Instruction:\nWrite a Python function to find the factorial of a number.\n\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Observations

- The fine-tuned model generates syntactically correct, runnable Python for standard algorithm tasks (factorial, Fibonacci, binary search, palindrome check, cycle detection).
- Performance degrades on conceptual/theory questions (e.g., time complexity explanation) — likely due to the dataset skewing heavily toward code-output problems.
- Generation speed is approximately 2.2–2.5 tokens/second on an A100.

## Security Note

The notebook contains a hardcoded Hugging Face token. **Do not commit tokens to version control.** Use environment variables or Colab Secrets instead:

```python
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])
```

## License

This project uses `mistralai/Mistral-7B-Instruct-v0.1`, which is subject to the [Mistral AI Research License](https://mistral.ai/news/mistral-7b/). Ensure compliance before commercial use.
