# tests/test_agent_base.py

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# 添加 src 目录到模块搜索路径，以便可以导入 src 目录中的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from agents.agent_base import AgentBase  # 导入要测试的 AgentBase类

class TestAgentBase(unittest.TestCase):

    @patch("builtins.open", new_callable=MagicMock)
    def setUp(self, mock_open):
        # 初始化文件读取模拟
        # 模拟prompt文件内容为有效文本
        mock_open.return_value.__enter__.return_value.read.side_effect = [
            "Test prompt",  # 用于 load_prompt
            '{"messages": ["Welcome"]}'  # 用于 load_intro
        ]
        # 初始化AgentBase实例
        self.agent = AgentBase(name="TestAgent", prompt_file="test_prompt.txt", intro_file="test_intro.json")

    @patch("builtins.open", new_callable=MagicMock)
    def test_load_prompt(self, mock_open):
        # 测试从文件加载系统提示语的功能
        mock_open.return_value.__enter__.return_value.read.return_value = "Test prompt"
        result = self.agent.load_prompt()
        self.assertEqual(result, "Test prompt")

    @patch("builtins.open", new_callable=MagicMock)
    def test_load_intro(self, mock_open):
        # 测试从 JSON 文件加载初始消息的功能
        mock_open.return_value.__enter__.return_value.read.return_value = '{"messages": ["Welcome"]}'
        result = self.agent.load_intro()
        self.assertEqual(result, {"messages": ["Welcome"]})

    @patch("agents.agent_base.ChatOllama")
    def test_create_chatbot(self, mock_chat_model):
        # 测试创建聊天机器人的功能
        self.agent.create_chatbot()
        mock_chat_model.assert_called_once()

    @patch("langchain_core.runnables.history.RunnableWithMessageHistory.invoke")
    def test_chat_with_history(self, mock_invoke):
        # 测试带有聊天历史的对话功能
        mock_invoke.return_value.content = "Test response"
        response = self.agent.chat_with_history("Hello")
        self.assertEqual(response, "Test response")

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_handle_missing_prompt_file(self, mock_open):
        # 测试提示文件不存在的情况
        with self.assertRaises(FileNotFoundError):
            self.agent.load_prompt()

    @patch("builtins.open", new_callable=MagicMock)
    def test_handle_invalid_intro_json(self, mock_open):
        # 测试初始消息文件的无效 JSON 情况
        mock_open.return_value.__enter__.return_value.read.return_value = "Invalid JSON"
        with self.assertRaises(ValueError):
            self.agent.load_intro()

    def tearDown(self):
        # 在每个测试之后执行的清理操作
        self.agent = None

if __name__ == '__main__':
    unittest.main()
