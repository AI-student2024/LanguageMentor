# tests/test_session_history.py

import sys
import os
import unittest

# 添加 src 目录到模块搜索路径，以便可以导入 src 目录中的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from agents.session_history import get_session_history, store  # 导入要测试的函数和存储对象
from langchain_core.messages import HumanMessage  # 用于创建测试消息

class TestSessionHistory(unittest.TestCase):

    def setUp(self):
        # 清空存储的会话历史，确保每个测试之间互不影响
        store.clear()

    def test_get_session_history_new_session(self):
        # 测试获取新的会话历史
        session_id = "test_session"
        history = get_session_history(session_id)
        self.assertEqual(len(history.messages), 0)

    def test_get_session_history_existing_session(self):
        # 测试获取已有的会话历史
        session_id = "test_session"
        store[session_id] = get_session_history(session_id)
        history = get_session_history(session_id)
        self.assertEqual(history, store[session_id])

    def test_session_history_persistence(self):
        # 测试会话历史的持久化
        session_id = "test_session"
        history = get_session_history(session_id)
        test_message = HumanMessage(content="Test message")
        history.add_message(test_message)
        self.assertEqual(store[session_id].messages[0].content, "Test message")

    def tearDown(self):
        # 清空存储的会话历史，确保测试之间互不影响
        store.clear()

if __name__ == '__main__':
    unittest.main()
