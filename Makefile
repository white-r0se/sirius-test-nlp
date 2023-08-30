build:
	docker-compose build

run:
	docker-compose up -d

stop:
	docker-compose down

tests:
	python3 -m pytest src/tests/bot_test.py
	python3 -m pytest src/tests/dataset_test.py
	python3 -m pytest src/tests/interface_test.py
	python3 -m pytest src/tests/finetune_test.py
	
clean:
	rm -rf output wandb models/rudialogpt-medium-lora-5ep-test