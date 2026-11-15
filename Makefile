.PHONY: test experiments

test:
	python3 -m pytest -q

experiments:
	python3 experiments/run_all.py
