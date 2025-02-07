change=patch
message=RELEASE


setup:
	pip install -r requirements.txt

setup_dev: setup
	pip install -r requirements.dev.txt

publish: lint
	@poetry version $(change)
	@git add .
	@git commit -m "$(message)"
	@git tag v$$(poetry version -s)
	@git push
	@git push origin --tags

lint:
	isort .
	black .
	flake8 .

test:
	pytest . -s

all: lint test
