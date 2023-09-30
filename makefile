.PHONY: test venv

run:
	python3 learn.py

test:
	python3 test.py

venv:
	source venv/bin/activate

build:
	sudo apt-get install python3.6
	sudo apt install python3-pip
	python3 -m venv venv
	source venv/bin/activate
	pip3 install numpy
	pip3 install opencv-python3
	pip3 install matplotlib
	pip3 install scipy
	pip3 install tensorflow
	sudo apt-get install unzip