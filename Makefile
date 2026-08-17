.PHONY: install train validate test lint fix-dockerignore preprod-up preprod-down preprod-ps preprod-logs smoke deploy rollback versions

COMPOSE_FILE := docker-compose.preprod.yml

install:
	pip install -r requirements.txt

train:
	python src/generate_data.py
	python src/train_pipeline.py

validate:
	python src/validate_model.py

test:
	pytest tests/ -v --ignore=tests/smoke

lint:
	flake8 api/ src/ tests/ --max-line-length=100

fix-dockerignore:
	@echo "El .dockerignore ya se llama .dockerignore en este repo (no requiere corrección)."

preprod-up:
	docker compose -f $(COMPOSE_FILE) up --build -d

preprod-down:
	docker compose -f $(COMPOSE_FILE) down -v

preprod-ps:
	docker compose -f $(COMPOSE_FILE) ps

preprod-logs:
	docker compose -f $(COMPOSE_FILE) logs -f

smoke:
	pytest tests/smoke/ -v --tb=short -q

deploy:
	bash deploy.sh $(VERSION)

rollback:
	docker tag prestamo-renovacion-api:$(VERSION) prestamo-renovacion-api:latest
	docker compose -f $(COMPOSE_FILE) up -d

versions:
	python src/manage_versions.py
