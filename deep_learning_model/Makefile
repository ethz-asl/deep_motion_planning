.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')

#################################################################################
# COMMANDS                                                                      #
#################################################################################

DATASET = shapes_300
DATASET_EVAL = shapes_300

requirements:
	pip install -q -r requirements.txt

data:
	python src/data/make_dataset.py --random data/mixers/data.mix $(DATASET).h5

train:
	python src/model/train_model.py data/processed/$(DATASET).h5 \
				 													data/processed/$(DATASET_EVAL).h5 \
				 													-l 0.0001 -s 2000000 -b 128 --mail

eval:
	python src/model/predict_model.py -c reports/evaluation.csv data/processed/$(DATASET_EVAL).h5 \
																		../deep_motion_planner/models/model.pb

clean:
	find . -name "*.pyc" -exec rm {} \;

tensorboard:
	tensorboard --logdir=models/default

