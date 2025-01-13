from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

load_dotenv()

class ENV:
    def __init__(self):
        pass
    
    def loading_variables_and_fucntions(self):
        tokenizer = AutoTokenizer.from_pretrained("path-to-tokenizer")
        model = AutoModelForSeq2SeqLM.from_pretrained("path-to-model")
        
        return tokenizer, model