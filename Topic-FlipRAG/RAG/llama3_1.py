from typing import Optional, List, Any
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from typing import Optional, List, Any
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLAMA3_1_LLM(LLM):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_name_or_path: str):
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("完成本地模型的加载")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[Any] = None, **kwargs: Any) -> str:
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([input_ids], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=512)
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response


    @property
    def _llm_type(self) -> str:
        return "LLAMA3_1_LLM"


