.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')

#################################################################################
# COMMANDS                                                                      #
#################################################################################

DATASET = eval_300
DATASET_EVAL = eval_200

# Makefile arguments
GPU_MACHINE = lynx
USERNAME = pfmark

requirements:
	pip install -q -r requirements.txt

data:
	python src/data/make_dataset.py --random data/mixers/data.mix $(DATASET).h5

train:
	python src/model/train_model.py data/processed/$(DATASET).h5 \
				 													data/processed/$(DATASET_EVAL).h5 \
				 													-l 0.0001 -s 200000 -b 128 --mail

eval:
	python src/model/predict_model.py -c reports/evaluation.csv data/processed/$(DATASET_EVAL).h5 \
																		../deep_motion_planner/models/model.pb
																		
sync-gpu: 
	rsync -av --progress --exclude models/ --exclude data/raw * $(USERNAME)@$(GPU_MACHINE):/home/$(USERNAME)/testing/imitation_learning/
	
train-gpu:
	rsync -av --progress --exclude models/ --exclude data/raw * $(USERNAME)@$(GPU_MACHINE):/home/$(USERNAME)/testing/imitation_learning/
	ssh -t $(USERNAME)@$(GPU_MACHINE) bash -il -c "'cd testing/imitation_learning; make train'"
	rm -rf model/*
	rsync -av --progress $(USERNAME)@$(GPU_MACHINE):/home/$(USERNAME)/testing/imitation_learning/models/default/* gpu_models/

clean:
	find . -name "*.pyc" -exec rm {} \;

tensorboard:
	tensorboard --logdir=models/default

