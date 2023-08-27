from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

__version__ = "0.0.1"

class InterfaceModel:
    """Interface model for the chatbot"""

    def __init__(self):
        self.model_path = "../models/rudialogpt-medium-lora-5ep"
        self.model = None
        self.tokenizer = None
        self.config = None
        self.chat_history = None
        if self.model is None:
            self.setup_model()

    def setup_model(self):
        self.config = PeftConfig.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(self.config.base_model_name_or_path)
        self.model = PeftModel.from_pretrained(model, self.model_path)
        self.chat_history = "@@ПЕРВЫЙ@@ "

    def predict(self, input_str: str) -> tuple[str]:
        """
        Predict the reply to the input text
        
        Args:
            input_str (str): input text
        
        Returns:
            tuple(str, str): tuple of the reply and updated chat history
        """
        if self.model is None:
            self.setup_model()
        self.chat_history = self.chat_history + input_str + " @@ВТОРОЙ@@ "
        # check max len seq
        if len(self.chat_history.split()) > 1000:
            self.chat_history = self.chat_history.split()[-1000:]
            self.chat_history = " ".join(self.chat_history)
        input_ids = self.tokenizer(self.chat_history, return_tensors="pt")
        generated_token_ids = self.model.generate(
            **input_ids,
            top_k=5,
            top_p=0.9,
            num_beams=3,
            num_return_sequences=1,
            do_sample=True,
            no_repeat_ngram_size=2,
            temperature=1.7,
            repetition_penalty=1.2,
            length_penalty=1.0,
            eos_token_id=50257,
            max_new_tokens=40,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_output = self.tokenizer.decode(generated_token_ids[0], skip_special_tokens=True)
        cutted_answer = generated_output[len(self.chat_history):]
        # cut off the answer if it contains the special tokens
        if "@@ПЕРВЫЙ@@" in cutted_answer:
            cutted_answer = cutted_answer.split("@@ПЕРВЫЙ@@")[0]
        if "@@ВТОРОЙ@@" in cutted_answer:
            cutted_answer = cutted_answer.split("@@ВТОРОЙ@@")[0]
        self.chat_history = self.chat_history + cutted_answer + " @@ПЕРВЫЙ@@ "
        return cutted_answer, self.chat_history
        
    def clear_history(self):
        """Clear the chat history"""
        self.chat_history = "@@ПЕРВЫЙ@@ "