# Specific-domain Generative-based Chatbot using Large Language Model

- Model : Llama 2 7b chat
- Dataset : [finance_questions by 4DR1455](https://huggingface.co/datasets/4DR1455/finance_questions)
- Maximum Length : 256
- Fine Tuning:
    * QLoRA Adapter on Transformer Attentions

- UI:
    * Streamlit simple chat UI
 

## Deployment

To deploy this project run

### Step 1 : Build Docker Image
```bash
  docker build -t finance-chatbot-app .
```

### Step 2 : Run Docker Container
```bash
  docker run -d --restart always --name finance-chatbot-app -p 8501:8501 finance-chatbot-app
```

To stop and delete the container run
```bash
  docker stop chatbot-app && docker rm finance-chatbot-app
```
