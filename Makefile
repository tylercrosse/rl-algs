.PHONY: install test reproduce synthetic

install:
	pip install -e .[dev]

synthetic:
	python scripts/run_reward_misspec_suite.py --synthetic

reproduce:
	python scripts/run_reward_misspec_suite.py

test:
	pytest
