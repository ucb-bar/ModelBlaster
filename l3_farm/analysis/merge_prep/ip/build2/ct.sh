#!/bin/bash
# ct.sh REPO MSGFILE PARENT  -> prints the sha of a commit object of the CURRENT INDEX (does not move any ref)
cd "$1" || exit 1
T=$(git write-tree) || exit 1
GIT_AUTHOR_NAME="Kris Dong" GIT_AUTHOR_EMAIL=chrisdong@berkeley.edu GIT_COMMITTER_NAME="Kris Dong" GIT_COMMITTER_EMAIL=chrisdong@berkeley.edu \
  git -c user.name="Kris Dong" -c user.email="chrisdong@berkeley.edu" commit-tree "$T" -p "$3" -F "$2"
