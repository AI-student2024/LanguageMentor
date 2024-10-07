import os
from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models import ChatOpenAI
from config import config

class ModelManager:
    def __init__(self, model_name=None):
        if not isinstance(model_name, str) or not model_name:
            self.model_name = config.get_default_model()
        else:
            self.model_name = model_name
        self.model_config = config.get_model_config(self.model_name)
        if not self.model_config:
            raise ValueError(f"模型配置未找到：{self.model_name}")
        self.model_type = self.model_config.get("type", config.get_default_model_type())

    def create_model(self):
        if self.model_type == "ollama":
            # Ollama 模型初始化
            return ChatOllama(
                model=self.model_name,
                max_tokens=self.model_config["max_tokens"],
                temperature=self.model_config["temperature"]
            )
        elif self.model_type == "openai":
            # 从环境变量中获取 API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key 未在环境变量中找到，请设置 `OPENAI_API_KEY` 环境变量。")
            # OpenAI 模型初始化
            return ChatOpenAI(
                model=self.model_name,
                api_key=api_key,
                max_tokens=self.model_config["max_tokens"],
                temperature=self.model_config["temperature"]
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
