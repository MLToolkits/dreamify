change=patch
message=RELEASE


setup:
	pip install -r requirements.txt

setup_dev: setup
	pip install -r requirements.dev.txt

publish:
	@poetry version $(change)
	@git tag v$$(poetry version -s)
	@git commit -am "$(message)"
	@git push origin --tags
	@poetry publish --build

lint:
	isort .
	black .
	flake8 .

all: lint 
