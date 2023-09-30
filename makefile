.PHONY: test venv

run:
	python3 learn.py

test:
	python3 test.py

build:
	sudo apt-get install python3.6
	sudo apt install python3-pip
	sudo apt-get install unzip
	python3 -m venv venv