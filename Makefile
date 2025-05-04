install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt &&\
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
	latest_png = $(ls -t ./results/model_results_*.png | head -n 1)
	echo '## Predicted vs Actual Scatter Plot' >> report.md
	echo '![Scatter Plot]($latest_png)' >> report.md

	cml comment create report.md

	

