PYTHON_FILES := $(shell find . -name '*.py')

build:
	docker-compose build

run:
	docker-compose up -d

stop:
	docker-compose down

test:
	python3 -m pytest tests/*.py
	
clean:
	rm -rf output wandb models/rudialogpt-medium-lora-5ep-test

lint:
	black $(PYTHON_FILES)