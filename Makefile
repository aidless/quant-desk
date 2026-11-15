# eval：跑测试套件（46 个用例，约 15 秒，不联网）。
.PHONY: eval

eval:
	python3 -m pytest tests -q
