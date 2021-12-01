check: check.style check.types check.units

check.style:
	flake8 ./phac ./examples

check.types:
	mypy --ignore-missing-imports ./phac

check.units:
	python -m pytest
