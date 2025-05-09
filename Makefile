# Makefile for the project

## Set the steps to run always in the pipeline
.PHONY: tests save_tests_report

## Linting
eval_docstyle:
	pip install --upgrade pip && \
	pip install pydocstyle && \
	pydocstyle --match=".*/*.py"

eval_codespell:
	pip install --upgrade pip && \
	pip install codespell && \
	codespell --skip="*.csv *.json *.png" \
	--skip="./notebook/*" --skip="./artifacts/* ./results/*" \
	--ignore-words=./codespell-ignore.txt

eval_isort:
	pip install --upgrade pip && \
	pip install isort && \
	isort --settings-path pyproject.toml --check-only --diff  .

eval_black:
	pip install --upgrade pip && \
	pip install black && \
	black --config pyproject.toml --check --diff --exclude "/(notebook|artifacts|results)/" .

eval_flake8:
	pip install --upgrade pip && \
	pip install flake8 flake8-pyproject && \
	flake8 --exclude notebook/,artifacts/,results/ .


## Tests 
install_tests:
	pip install --upgrade pip && \
    pip install -r requirements.txt && \
	pip install -e . && \
    pip install pytest pytest-cov

tests:
	pytest --cov=src --cov-report=term-missing --cov-report=html && \
	coverage html

save_tests_report:
	coverage html && \
    ls -R htmlcov/


## CML 
install_cml:
	pip install --upgrade pip &&\
	pip install -r requirements.txt  &&\
	pip install -e .

format:
	black *.py

train:
	python ./src/pipeline/train_pipeline.py

eval:
	echo "## Model Metrics" > report.md
#   print the txt content 
#	cat ./results/metrics.txt >> report.md
#   print the last line of the txt file
	tail -n 1 ./results/metrics.txt >> report.md

#   print the last png file in the results folder
	latest_png=$$(ls -t ./results/model_results_*.png | head -n 1); \
	echo "\n## Predicted vs Actual Scatter Plot" >> report.md; \
	echo "The latest PNG is: $${latest_png}" >> report.md; \
	echo "\n![Scatter Plot]($${latest_png})" >> report.md

	cml comment create report.md

# update branch
update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin HEAD:update

