# tests/test_scenario_agent.py

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# 添加 src 目录到模块搜索路径，以便可以导入 src 目录中的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from agents.scenario_agent import ScenarioAgent  # 导入要测试的 ScenarioAgent 类

class TestScenarioAgent(unittest.TestCase):

    @patch("builtins.open", new_callable=MagicMock)
    def setUp(self, mock_open):
        # 模拟 prompt_file 和 intro_file 的文件内容
        mock_open.return_value.__enter__.return_value.read.side_effect = [
            "Test scenario prompt",  # 用于 prompt_file
            '{"messages": ["Scenario intro message"]}'  # 用于 intro_file
        ]
        # 初始化 ScenarioAgent 实例
        self.agent = ScenarioAgent(scenario_name="test_scenario", session_id="test_session")

    @patch("builtins.open", new_callable=MagicMock)
    def test_load_prompt(self, mock_open):
        # 测试从文件加载系统提示语的功能
        mock_open.return_value.__enter__.return_value.read.return_value = "Test scenario prompt"
        result = self.agent.load_prompt()
        self.assertEqual(result, "Test scenario prompt")

    @patch("builtins.open", new_callable=MagicMock)
    def test_load_intro(self, mock_open):
        # 测试从 JSON 文件加载初始消息的功能
        mock_open.return_value.__enter__.return_value.read.return_value = '{"messages": ["Scenario intro message"]}'
        result = self.agent.load_intro()
        self.assertEqual(result, {"messages": ["Scenario intro message"]})

    @patch("agents.agent_base.ChatOllama")
    def test_create_chatbot(self, mock_chat_model):
        # 模拟 ChatOllama
        mock_instance = MagicMock()
        mock_chat_model.return_value = mock_instance

        # 调用 create_chatbot 方法
        self.agent.create_chatbot()

        # 确保 ChatOllama 被调用一次
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

    @patch("agents.session_history.get_session_history")
    @patch("random.choice", return_value="Initial AI message")
    def test_start_new_session(self, mock_random_choice, mock_get_session_history):
        # 测试启动新会话的功能
        mock_get_session_history.return_value.messages = []  # 假设历史为空
        initial_message = self.agent.start_new_session(session_id="test_session")
        self.assertEqual(initial_message, "Initial AI message")
        mock_random_choice.assert_called_once_with(self.agent.intro_messages)

    def tearDown(self):
        # 在每个测试之后执行的清理操作
        self.agent = None

if __name__ == '__main__':
    unittest.main()
