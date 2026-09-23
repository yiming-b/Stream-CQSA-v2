#!/bin/bash
# Freeze the main repo's current commit for an OOC-rivals slurm job; prints the snapshot directory.
#   SNAP=$(bash slurm/ooc_rivals/snapshot.sh) && sbatch --export=ALL,CQSA_SRC=$SNAP slurm/ooc_rivals/<job>.slurm
# A queued job that imports the live tree runs whatever is checked out when it starts; the snapshot
# (a `git archive` of one commit, so uncommitted edits are NOT included) cannot change. Results and
# logs still go to the live tree (results/ooc_rivals/), whose files the jobs only append to.
set -euo pipefail
REPO=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-v2
SHA=$(git -C "$REPO" rev-parse --short HEAD)
SNAP=/scratch/gpfs/AKEY/yb2807/Stream-CQSA-snapshots/v2-$SHA
if [ ! -f "$SNAP/.complete" ]; then
  mkdir -p "$SNAP"
  git -C "$REPO" archive HEAD | tar -x -C "$SNAP"
  touch "$SNAP/.complete"
fi
echo "$SNAP"
