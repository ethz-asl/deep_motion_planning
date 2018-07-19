.PHONY: requirements data train eval sync-gpu train-gpu clean tensorboard

#################################################################################
# GLOBALS                                                                       #
#################################################################################

DATASET = office_1000
DATASET_EVAL = eval_300

# Makefile arguments
GPU_MACHINE = hyrax
USERNAME = pfmark

# Training parameters
LEARNING_RATE = 0.0001
TRAINING_STEPS = 1000000
BATCH_SIZE = 128

#################################################################################
# COMMANDS                                                                      #
#################################################################################

data:
	python src/data/make_dataset.py --random data/mixers/data.mix $(DATASET).h5

train:
	python src/model/train_model.py data/processed/$(DATASET).h5 \
				 													data/processed/$(DATASET_EVAL).h5 \
				 													-l $(LEARNING_RATE) -s $(TRAINING_STEPS) -b $(BATCH_SIZE) --mail

eval:
	python src/model/predict_model.py -c reports/evaluation.csv data/processed/$(DATASET_EVAL).h5 \
																		../deep_motion_planner/models/model.pb
																		
sync-gpu: 
	rsync -av --progress --exclude models/ --exclude data/raw * $(USERNAME)@$(GPU_MACHINE):/home/$(USERNAME)/training/imitation_learning/
	
clean:
	find . -name "*.pyc" -exec rm {} \;

tensorboard:
	tensorboard --logdir=models/default

