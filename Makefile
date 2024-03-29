
install:
	pip install -r requirements.txt --upgrade
	pip install -r requirements_dev.txt --upgrade
	pip install -e .
	pre-commit install

test:
	pytest

cov:
	pytest --cov= emepy

mypy:
	mypy . --ignore-missing-imports

lint:
	flake8

pylint:
	pylint emepy

lintd2:
	flake8 --select RST

lintd:
	pydocstyle emepy

doc8:
	doc8 docs/

update:
	pur

update2:
	pre-commit autoupdate --bleeding-edge
