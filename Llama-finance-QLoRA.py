import re
import gc

from datasets import load_dataset
from transformers import get_scheduler, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

use_4bit = True
bnb_4bit_compute_dtype = "float16"
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

def inference(query, model, tokenizer):
    query = f"Answer this following question:\n{query}{tokenizer.eos_token}"
    inputs = tokenizer.encode(query, return_tensors='pt').to(device)
    response = model.generate(
        inputs.to(device), 
        max_length=1000, 
        do_sample=True,
        top_k=15, 
        top_p=0.9, 
        temperature=0.5,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(response[0], skip_special_tokens=False)
    return response.split(tokenizer.eos_token)[1]

# link dataset : https://huggingface.co/datasets/4DR1455/finance_questions?row=3
dataset = load_dataset("4DR1455/finance_questions")
dataset["train"] = dataset["train"].select(range(1280))

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

original_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-chat-hf',
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config, 
    device_map=device
)

original_model.config.use_cache = False
original_model.config.pretraining_tp = 1

config = LoraConfig(
    r=16, #Rank
    lora_alpha=32,
    target_modules=[
        'q_proj',
        'k_proj',
        'v_proj',
    ],
    bias="none",
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
)


model = get_peft_model(original_model, config)

def preprocess_function(examples):
    questions = [ex['instruction'] for ex in examples]
    answers = [ex['output'] for ex in examples]
    inputs = [f"Answer this following question:\n{q}\nAnswer:\n{a}{tokenizer.eos_token}" for q, a in zip(questions, answers)]
    model_inputs = tokenizer(inputs, max_length=256, padding=True, truncation=True, return_tensors="pt")
    return model_inputs

train_dataloader = DataLoader(dataset['train'], batch_size=4, collate_fn=preprocess_function)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 20
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    
    train_loss = 0
    progress_bar = tqdm(train_dataloader, desc="Training", leave=True)

    print()
    print("Inference:")
    print(inference("How do dividend policies impact a company's financial performance?", model, tokenizer))
    print()

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        train_loss += loss.item()
        
        progress_bar.set_postfix({"Loss": loss.item()})
        
        # Memory Optimization
        del batch
        del loss
        del input_ids
        del attention_mask
        del labels
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        
    avg_train_loss = train_loss / len(train_dataloader)
    print(f"Training loss: {avg_train_loss:.4f}")

    model.save_pretrained("./fine-tuned-llama2-7b-chat-hf-aistudio-finance-QLoRA3-subset")
    tokenizer.save_pretrained("./fine-tuned-llama2-7b-chat-hf-aistudio-finance-QLoRA3-subset")