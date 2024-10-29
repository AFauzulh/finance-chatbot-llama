import streamlit as st

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login


login(token="YOUR_HUGGINGFACE_TOKEN")


if torch.cuda.is_available():       
    device = torch.device("cuda")
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

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

if "model" not in st.session_state.keys():
    st.session_state["model"] = AutoModelForCausalLM.from_pretrained(
                                './fine-tuned-llama2-7b-chat-hf-aistudio-finance-QLoRA3-subset',
                                cache_dir="./model_cache",
                                torch_dtype=torch.bfloat16,
                                quantization_config=bnb_config, 
                                device_map=device 
    )

if "tokenizer" not in st.session_state.keys():
    st.session_state["tokenizer"] = AutoTokenizer.from_pretrained('./fine-tuned-llama2-7b-chat-hf-aistudio-finance-QLoRA3-subset')

def inference_qlora_prompt(query, model, tokenizer):
    query = f"Answer this following question:\n{query}{tokenizer.eos_token}"
    inputs = tokenizer.encode(query, return_tensors="pt").to(device)
 
    response = model.generate(
        inputs.to(device), 
        max_length=1000, 
        do_sample=True,
        top_k=15, 
        top_p=0.9, 
        temperature=0.5,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(response[0], skip_special_tokens=True)

st.title("💬 Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.spinner("generating response. . ."):
        response = inference_qlora_prompt(st.session_state.messages[-1]['content'], st.session_state["model"], st.session_state["tokenizer"])

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)