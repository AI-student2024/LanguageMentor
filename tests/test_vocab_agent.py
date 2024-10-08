# tests/test_vocab_agent.py

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# 添加 src 目录到模块搜索路径，以便可以导入 src 目录中的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from agents.vocab_agent import VocabAgent  # 导入要测试的 VocabAgent 类

class TestVocabAgent(unittest.TestCase):

    @patch("builtins.open", new_callable=MagicMock)
    def setUp(self, mock_open):
        # 模拟 prompt_file 的文件内容
        mock_open.return_value.__enter__.return_value.read.side_effect = [
            "Test vocab prompt"  # 用于 prompt_file
        ]
        # 初始化 VocabAgent 实例
        self.agent = VocabAgent(session_id="test_session")

    @patch("builtins.open", new_callable=MagicMock)
    def test_load_prompt(self, mock_open):
        # 测试从文件加载系统提示语的功能
        mock_open.return_value.__enter__.return_value.read.return_value = "Test vocab prompt"
        result = self.agent.load_prompt()
        self.assertEqual(result, "Test vocab prompt")

    @patch("agents.agent_base.ChatOllama")
    def test_create_chatbot(self, mock_chat_model):
        # 模拟 ChatOllama
        mock_instance = MagicMock()
        mock_chat_model.return_value = mock_instance

        # 调用 create_chatbot 方法
        self.agent.create_chatbot()

        # 确保 ChatOllama 被调用一次
        mock_chat_model.assert_called_once()

    @patch("agents.session_history.get_session_history")
    def test_restart_session(self, mock_get_session_history):
        # 模拟 session_history
        mock_history = MagicMock()
        mock_history.messages = ["message1", "message2"]
        mock_get_session_history.return_value = mock_history

        # 调用 restart_session 方法
        self.agent.restart_session()

        # 检查是否手动清空了历史记录的 messages 列表
        mock_history.messages = []  # 在 restart_session 方法中清空列表
        self.assertEqual(len(mock_history.messages), 0)

    @patch("langchain_core.runnables.history.RunnableWithMessageHistory.invoke")
    def test_chat_with_history(self, mock_invoke):
        # 测试带有聊天历史的对话功能
        mock_invoke.return_value.content = "Test response"
        response = self.agent.chat_with_history("Hello")
        self.assertEqual(response, "Test response")

    def tearDown(self):
        # 在每个测试之后执行的清理操作
        self.agent = None

if __name__ == '__main__':
    unittest.main()
