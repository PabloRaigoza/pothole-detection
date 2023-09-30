.PHONY: test venv

run:
	venv/bin/python3 python3 learn.py

test:
	venv/bin/python3 test.py

build:
	sudo apt-get install python3.6
	sudo apt install python3-pip
	sudo apt-get install unzip
	unzip archive.zip
	mkdir test; mv normal test/normal; mv potholes test/potholes
	python3 -m venv venv
	venv/bin/pip3 install -r requirements.txt