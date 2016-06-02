.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')

#################################################################################
# COMMANDS                                                                      #
#################################################################################

RAW_DATASET = 2016-05-27_19-18-40
DATASET = test

requirements:
	pip install -q -r requirements.txt

data:
	python src/data/make_dataset.py data/raw/$(RAW_DATASET)/ $(DATASET).h5

train:
	python src/model/train_model.py data/processed/$(DATASET).h5

clean:
	find . -name "*.pyc" -exec rm {} \;

lint:
	flake8 --exclude=lib/,bin/ .

sync_data_to_s3:
	aws s3 sync data/ s3://$(BUCKET)/data/

sync_data_from_s3:
	aws s3 sync s3://$(BUCKET)/data/ data/

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
