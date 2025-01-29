change=patch
message=RELEASE


setup:
	pip install -r requirements.txt

setup_dev: setup
	pip install -r requirements.dev.txt

publish:
	@poetry version $(change)
	@version=$$(poetry version -s) && \
	poetry build && \
	git tag v$$version && \
	git add . && \
    git commit -m "$(message)" && \
	git push origin v$$version

lint:
	isort .
	black .
	flake8 .

all: lint 