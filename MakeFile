install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

format:
	black *.py

train:
	python src/pipeline/train.py

eval:
	echo "## Model Metrics" > report.md
#	cat ./results/metrics.txt >> report.md
	tail -n 1 ./results/metrics.txt >> report.md

	latest_png = $(ls -t ./results/model_results_*.png | head -n 1)
	echo '## Predicted vs Actual Scatter Plot' >> report.md
	echo '![Scatter Plot](latest_png)' >> report.md

	cml comment create report.md

	

