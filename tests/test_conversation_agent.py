# tests/test_conversation_agent.py
# tests/test_conversation_agent.py

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# 添加 src 目录到模块搜索路径，以便可以导入 src 目录中的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from agents.conversation_agent import ConversationAgent  # 导入要测试的 ConversationAgent 类

class TestConversationAgent(unittest.TestCase):
    
    def setUp(self):
        # 初始化 ConversationAgent 对象，在每个测试之前都会执行
        self.agent = ConversationAgent(session_id="test_session")

    @patch("builtins.open", new_callable=MagicMock)
    def test_load_prompt(self, mock_open):
        # 测试从文件加载系统提示语的功能
        mock_open.return_value.__enter__.return_value.read.return_value = "Test conversation prompt"
        result = self.agent.load_prompt()
        self.assertEqual(result, "Test conversation prompt")

    @patch("langchain_core.runnables.history.RunnableWithMessageHistory.invoke")
    def test_chat_with_history(self, mock_invoke):
        # 测试带有聊天历史的对话功能
        mock_invoke.return_value.content = "Test response"
        response = self.agent.chat_with_history("Hello")
        self.assertEqual(response, "Test response")

    def test_handle_missing_prompt_file(self):
        # 测试提示文件不存在的情况
        self.agent.prompt_file = "missing_conversation_prompt.txt"
        with self.assertRaises(FileNotFoundError):
            self.agent.load_prompt()

    def tearDown(self):
        # 在每个测试之后执行的清理操作
        self.agent = None

if __name__ == '__main__':
    unittest.main()