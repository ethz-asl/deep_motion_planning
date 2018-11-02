.PHONY: requirements data train eval sync-gpu train-gpu clean tensorboard

#################################################################################
# GLOBALS                                                                       #
#################################################################################

DATASET = shapes_1000
DATASET_EVAL = eval_300

# Makefile arguments
GPU_MACHINE = hyrax
USERNAME = pfmark

# Training parameters
LEARNING_RATE = 0.0001
TRAINING_STEPS = 100000
BATCH_SIZE = 128

#################################################################################
# COMMANDS                                                                      #
#################################################################################

requirements:
	pip install -q -r requirements.txt

data:
	python src/data/make_dataset.py --random data/mixers/data.mix $(DATASET).h5

train:
	python src/model/train_model.py data/processed/$(DATASET).h5 \
				 													-l $(LEARNING_RATE) -s $(TRAINING_STEPS) -b $(BATCH_SIZE) --mail

eval:
	python src/model/predict_model.py -c reports/evaluation.csv data/processed/$(DATASET_EVAL).h5 \
																		../deep_motion_planner/models/model.pb
																		
sync-gpu: 
	rsync -av --progress --exclude models/ --exclude data/raw * $(USERNAME)@$(GPU_MACHINE):/home/$(USERNAME)/training/imitation_learning/
	
train-gpu:
	rsync -av --progress --exclude models/ --exclude data/raw * $(USERNAME)@$(GPU_MACHINE):/home/$(USERNAME)/training/imitation_learning/
	ssh -t $(USERNAME)@$(GPU_MACHINE) bash -il -c "'cd testing/imitation_learning; make train'"
	rm -rf model/*
	rsync -av --progress $(USERNAME)@$(GPU_MACHINE):/home/$(USERNAME)/testing/imitation_learning/models/default/* gpu_models/

clean:
	find . -name "*.pyc" -exec rm {} \;

tensorboard:
	tensorboard --logdir=models/default

