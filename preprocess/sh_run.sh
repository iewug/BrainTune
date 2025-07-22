#!/bin/bash

##############################
# Run in parallel            #
# Reports errors if any occur #
#############################

# Name of the script to run
# PLEASE CHANGE FILE NAME HERE
PYTHON_SCRIPT="buildArtifact"

# Define an array to hold the parameters
names=()
# Double loop to generate array; too many elements might cause memory issues
for i in {1..15}; do
  for j in {1..3}; do
    names+=("${i}_${j}")
  done
done

# Associative array to store PID and corresponding name for each background process
declare -A pid_names

# Iterate through the array and execute commands in parallel for each element
for name in "${names[@]}"; do
  python "${PYTHON_SCRIPT}.py" --name "$name" &
  pid=$!
  pid_names[$pid]=$name
done

# waiting for all background processes to complete and checking exit status
for pid in "${!pid_names[@]}"; do
  wait "$pid"
  exit_status=$?
  if [ $exit_status -ne 0 ]; then
    echo "The program 'python ${PYTHON_SCRIPT}.py --name ${pid_names[$pid]}' encountered an error, exit status: $exit_status"
  fi
done

