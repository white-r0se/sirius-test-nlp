build:
	docker-compose build

run:
	docker-compose up -d

stop:
	docker-compose down

tests:
	python3 src/tests/bot_test.py
	python3 src/tests/interface_test.py
	python3 src/tests/finetune_test.py

clean:
	rm -rf output wandb models/rudialogpt-medium-lora-5ep-test