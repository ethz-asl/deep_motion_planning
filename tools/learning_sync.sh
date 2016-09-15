#!/bin/bash

THIS_PATH=$(pwd)/$(dirname $0)
REMOTE_USER=<user>
REMOTE_HOST=129.132.39.88

rsync -avz --exclude-from=$THIS_PATH/rsync_exclude -r $THIS_PATH/../deep_learning_model $REMOTE_USER@$REMOTE_HOST:
