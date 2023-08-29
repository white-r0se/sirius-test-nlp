build:
	docker-compose build

run:
	docker-compose up -d

stop:
	docker-compose down

tests:
	python3 -m src.tests.bot_test
	python3 -m src.tests.dataset_test
	python3 -m src.tests.interface_test
	python3 -m src.tests.finetune_test

clean:
	rm -rf output wandb models/rudialogpt-medium-lora-5ep-test