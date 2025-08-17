PY=python -m
RUN=$(PY) streamlit run streamlit_app.py
RUFF=ruff
PORT?=8501
BIND?=localhost

.PHONY: run run-docker lint fix test ci

run:
	$(RUN) --server.port $(PORT) --server.address $(BIND)
run-docker:
	docker compose up --build
lint:
	$(RUFF) check .
fix:
	$(RUFF) check --fix
test:
	pytest -q || true
ci: clean lint test
clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	$(RUFF) clean || true

fmt:
	ruff format .

fix:
	ruff check --select I,UP --fix .
	ruff format .
