.PHONY: test venv

run:
	python3 learn.py

test:
	python3 test.py

venv:
	source venv/bin/activate