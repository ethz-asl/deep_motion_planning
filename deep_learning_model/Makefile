.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')

#################################################################################
# COMMANDS                                                                      #
#################################################################################

DATASET = office_simple_borderless_filtered
DATASET_EVAL = evaluation

requirements:
	pip install -q -r requirements.txt

data:
	python src/data/make_dataset.py --random data/mixers/data.mix $(DATASET).h5

train:
	python src/model/train_model.py data/processed/$(DATASET).h5 \
				 													data/processed/$(DATASET_EVAL).h5 \
				 													-l 0.00001 -s 1000000 -b 128 --mail

eval:
	python src/model/predict_model.py -c reports/evaluation.csv data/processed/$(DATASET_EVAL).h5 \
																		../deep_motion_planner/models/model.pb

clean:
	find . -name "*.pyc" -exec rm {} \;

lint:
	flake8 --exclude=lib/,bin/ .

sync_data_to_s3:
	aws s3 sync data/ s3://$(BUCKET)/data/

sync_data_from_s3:
	aws s3 sync s3://$(BUCKET)/data/ data/

tensorboard:
	tensorboard --logdir=models/default

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
