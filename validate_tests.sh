#!/bin/bash

# 运行单元测试并将结果实时输出到 test_results.txt 和终端
python -m unittest discover -s tests -p "test_*.py" | tee test_results.txt

# 检查测试结果，如果有失败或错误，输出失败信息并让脚本退出状态为 1
if grep -q "FAILED" test_results.txt || grep -q "ERROR" test_results.txt; then
    echo "Some tests failed. Please check the details above."
    exit 1
else
    echo "All tests passed!"
    exit 0
fi
