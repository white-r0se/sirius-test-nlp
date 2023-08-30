build:
	docker-compose build

run:
	docker-compose up -d

stop:
	docker-compose down

test:
	python3 -m pytest tests/bot_test.py
	python3 -m pytest tests/dataset_test.py
	python3 -m pytest tests/interface_test.py
	python3 -m pytest tests/finetune_test.py
	
clean:
	rm -rf output wandb models/rudialogpt-medium-lora-5ep-test