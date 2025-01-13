# DeepLearning-GenAI

Welcome to the **DeepLearning-GenAI** repository! This is a repository dedicated to implementing various deep learning and generative AI projects. The aim is to explore cutting-edge algorithms and create solutions for a range of problems using the latest advancements in artificial intelligence and machine learning.

---

## Project: Text Summarization

This project focuses on developing a machine learning model for text summarization. The implementation leverages a preprocessed dataset to train a model capable of generating concise and meaningful summaries from input text.
### Dataset:
The dataset used to train the model is https://huggingface.co/datasets/Samsung/samsum
### Model
The model employed for text summarization is a transformer-based architecture (e.g., Pegasus). This model is fine-tuned on the provided dataset to optimize their summarization capabilities.
The training is configured using Hugging Face's `TrainingArguments` and `Trainer` classes.

```python
from transformers import TrainingArguments, Trainer

trainer_args = TrainingArguments(
    output_dir='pegasus-samsum', num_train_epochs=1, warmup_steps=500,
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    weight_decay=0.01, logging_steps=10,
    evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
    gradient_accumulation_steps=16
)
```
[Training must be done on training data for more epochs to have robust model]
### Model Saving
The trained model and its tokenizer are saved in the following directory:
```bash
./pegasus-samsum/
```
This includes:
- `pytorch_model.bin`: The model weights
- `config.json`: The configuration file
- `tokenizer/`: The tokenizer files (e.g., vocab.json, merges.txt)

### Inference
To use the trained model for summarization:
1. Load the model and tokenizer:
   ```python
   from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

   model_path = './pegasus-samsum/'
   model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   ```
2. Prepare the input text and generate the summary:
   ```python
   input_text = "Your input text here."
   inputs = tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=512)

   # Generate summary
   summary_ids = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
   summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

   print("Summary:", summary)
   ```

### Requirements
- Python 3.7+
- PyTorch
- Transformers library (Hugging Face)
- Additional libraries: numpy, pandas, scikit-learn

Install dependencies using:
```bash
pip install -r requirements.txt
```

### Conclusion
This project demonstrates the ability to fine-tune and deploy a transformer-based model for text summarization, providing users with concise representations of large texts. Feel free to customize the model and dataset for your use case!
