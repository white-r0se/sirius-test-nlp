build:
	docker-compose build

run:
	docker-compose up -d

stop:
	docker-compose down

tests:
	python3 src/tests/bot_test.py
	python3 src/tests/interface_test.py