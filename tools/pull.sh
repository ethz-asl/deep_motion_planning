#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: ./pull.sh <model>"
  echo "Example: ./pull.sh 2016-08-05_22-02_convolutional"
  exit 1
fi
echo ${1}

THIS_PATH=$(pwd)/$(dirname $0)
REMOTE_USER=<user>
REMOTE_HOST=129.132.39.88

scp -r $REMOTE_USER@$REMOTE_HOST:deep_learning_model/models/default/${1}/model.pb $THIS_PATH/../deep_motion_planner/models/

