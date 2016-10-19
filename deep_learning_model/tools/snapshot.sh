#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: ./snapshot.sh <model>"
  echo "Example: ./snapshot.sh 2016-08-05_22-02_convolutional/"
  exit 1
fi
echo ${1}

THIS_PATH=$(pwd)/$(dirname $0)

TARGET_PATH=$THIS_PATH/../models/default/

rm -rf $TARGET_PATH/snapshot/*
mkdir -p $TARGET_PATH/snapshot

cp -r $TARGET_PATH/${1}/checkpoint $TARGET_PATH/snapshot
cp -r $TARGET_PATH/${1}/graph.pb $TARGET_PATH/snapshot

SNAPSHOT=$(echo $(head -n 1 $TARGET_PATH/snapshot/checkpoint) | awk -F'"' '$0=$2')

cp -r $TARGET_PATH/${1}/$SNAPSHOT $TARGET_PATH/snapshot

