# IBM_Granite_llm

A hands-on repository for experimenting with **IBM Granite language models** using **Google Colab** and **Hugging Face**.
This project is built for learning, testing, and developing real-world applications powered by Granite LLMs.

---

## 🚀 What’s Inside

* Granite model setup on Google Colab
* Chatbot implementation using Granite 4.0 Micro
* Prompt engineering and testing examples
* Code generation and text summarization experiments
* Easy-to-follow notebooks for beginners

---

## 🧠 About Granite

IBM Granite is a family of enterprise-ready large language models designed for:

* Text generation
* Coding assistance
* Question answering
* Summarization
* Structured reasoning

This repo focuses on running Granite locally on free Colab GPUs.

---

## ⚙️ Quick Start (Google Colab)

### 1. Enable GPU

Runtime → Change runtime type → **GPU**

### 2. Install Dependencies

```python
!pip install -q transformers accelerate torch
```

### 3. Load Granite Model

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "ibm-granite/granite-4.0-micro"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
```

### 4. Run a Chat Prompt

```python
messages = [{"role": "user", "content": "Who are you?"}]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(inputs, max_new_tokens=60)

print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```

---

## 💬 Chatbot Mode

```python
def granite_chat():
    history = []
    print("Chat started. Type 'exit' to stop.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        history.append({"role": "user", "content": user_input})

        inputs = tokenizer.apply_chat_template(
            history,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(inputs, max_new_tokens=150)

        reply = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        print("Granite:", reply)

        history.append({"role": "assistant", "content": reply})

granite_chat()
```

---

## 📁 Repository Structure

```
IBM_Granite_llm/
├── notebooks/
│   ├── granite_chatbot.ipynb
│   ├── granite_summarizer.ipynb
│
├── examples/
│   ├── prompts.txt
│   ├── sample_outputs.md
│
└── README.md
```

---

## 🎯 Goals of This Repo

* Learn how Granite models work
* Build simple LLM-powered applications
* Experiment with prompts and responses
* Share Colab-based AI projects

---

## 📌 Notes

* This repo uses **Granite 4.0 Micro** because it runs on free Colab GPUs
* Larger Granite models require more VRAM or Hugging Face authentication

---

## 📖 License

This repository is for **educational and research purposes** only.
Please follow IBM Granite model licensing when using the models.

---

Happy experimenting with Granite! 💎

