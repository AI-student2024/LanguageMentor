import json
import os

class Config:
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "r", encoding="utf-8") as file:
                return json.load(file)
        else:
            raise FileNotFoundError(f"配置文件 {self.config_file} 不存在！")

    def get_default_model(self):
        return self.config["default_model"]

    def get_default_model_type(self):
        return self.config["default_model_type"]

    def get_model_config(self, model_name=None):
        if not model_name:
            model_name = self.get_default_model()
        return self.config["available_models"].get(model_name, None)

# 创建全局的配置管理器实例
config = Config()
