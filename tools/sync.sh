#!/bin/bash

REMOTE_USER=<user>
REMOTE_HOST=<host>

rsync -avz --exclude-from=./tools/rsync_exclude --port 22 ../deep_motion_planning \
  $REMOTE_USER@$REMOTE_HOST:

