"""
LLM 客户端：加载本地模型并生成文本
支持 HuggingFace Transformers 格式的开源模型（如 Qwen, ChatGLM, Llama 等）
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList
)
from typing import Optional


class StopOnTokens(StoppingCriteria):
    """自定义停止条件：遇到 EOS 或特定 token 停止"""
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class LLMClient:
    _instance = None  # 单例

    def __new__(cls, model_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: Optional[str] = None):
        if self._initialized:
            return

        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "models",
                "qwen",
                "Qwen2-1___5B-Instruct"   # ← 注意这里是三个下划线
            )

        #print(f"🧠 正在加载 LLM 模型: {model_path}")
        
        self.device = "cuda" if torch.cuda.is_available() else \
                      "mps" if torch.backends.mps.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        ).to(self.device) if self.device != "cuda" else AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.model.eval()
        #print(f"✅ LLM 模型加载完成，运行设备: {self.device}")
        self._initialized = True

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        生成回答
        
        Args:
            prompt: 完整提示词
            max_new_tokens: 最大生成长度
        
        Returns:
            生成的回答文本（不含 prompt）
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 设置停止条件（防止无限生成）
        stop_token_ids = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, 'im_end_id'):
            stop_token_ids.append(self.tokenizer.im_end_id)  # Qwen 特有

        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()