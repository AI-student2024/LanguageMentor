import json
import random

from langchain_ollama.chat_models import ChatOllama  # 导入 ChatOllama 模型
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 导入提示模板相关类
from langchain_core.messages import HumanMessage, AIMessage  # 导入人类消息类和AI消息类
from langchain_core.runnables.history import RunnableWithMessageHistory  # 导入带有消息历史的可运行类

from .session_history import get_session_history  # 导入会话历史相关方法
from utils.logger import LOG

from .model import ModelManager  # 导入 ModelManager

class ScenarioAgent:
    def __init__(self, scenario_name, model_name=None):
        self.name = scenario_name
        self.prompt_file = f"prompts/{self.name}_prompt.txt"
        self.intro_file = f"content/intro/{self.name}.json"
        self.prompt = self.load_prompt()
        self.intro_messages = self.load_intro()

        # 使用 ModelManager 获取模型
        model_manager = ModelManager(model_name=model_name)
        self.model = model_manager.create_model()

        self.create_chatbot()

    def load_prompt(self):
        try:
            with open(self.prompt_file, "r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file {self.prompt_file} not found!")

    def load_intro(self):
        try:
            with open(self.intro_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Intro file {self.intro_file} not found!")
        except json.JSONDecodeError:
            raise ValueError(f"Intro file {self.intro_file} contains invalid JSON!")
        

    def set_model(self, model):
        self.model = model
        self.create_chatbot()  # 重新初始化聊天机器人
        

    def create_chatbot(self):
        # 创建聊天提示模板，包括系统提示和消息占位符
        system_prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt),  # 系统提示部分
            MessagesPlaceholder(variable_name="messages"),  # 消息占位符
        ])

        # 使用 ModelManager 创建的模型
        self.chatbot = system_prompt | self.model

        # 将聊天机器人与消息历史记录关联起来
        self.chatbot_with_history = RunnableWithMessageHistory(self.chatbot, get_session_history)

    def start_new_session(self, session_id: str = None):
        if session_id is None:
            session_id = self.name

        history = get_session_history(session_id)
        LOG.debug(f"[history]:{history}")

        if not history.messages:  # 检查历史记录是否为空
            initial_ai_message = random.choice(self.intro_messages)  # 随机选择初始AI消息
            history.add_message(AIMessage(content=initial_ai_message))  # 添加初始AI消息到历史记录
            return initial_ai_message
        else:
            return history.messages[-1].content  # 返回历史记录中的最后一条消息

    def chat_with_history(self, user_input, session_id: str = None):
        if session_id is None:
            session_id = self.name

        response = self.chatbot_with_history.invoke(
            [HumanMessage(content=user_input)],  # 将用户输入封装为 HumanMessage
            {"configurable": {"session_id": session_id}},  # 传入配置，包括会话ID
        )
        
        return response.content  # 返回生成的回复内容
