#!/bin/bash
# set current directory of the script
# Get the directory path of the script
base_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "base_script_dir: ${base_script_dir}"

# Change the working directory to the script's directory
cd $base_script_dir

# Set default values only if not already set in the environment
# export DATASET_PATH=${DATASET_PATH:-/home/ron/Documents/smart-home/faces-train}
# export DATASET_PATH=${DATASET_PATH:-/home/ron/Documents/smart-home/faces-train/errors}
# export DATASET_PATH=${DATASET_PATH:-/home/ron/Documents/smart-home/faces-train/lfw_funneled}
export DATASET_PATH=${DATASET_PATH:-/home/ron/Documents/smart-home/faces-train/known}
export DOUBLE_TAKE_URL=${DOUBLE_TAKE_URL:-http://localhost:3000}
# export DOUBLE_TAKE_URL=${DOUBLE_TAKE_URL:-https://123c4afc1035a38b3690c92f7c387403.m.pipedream.net}
export COMPREFACE_URL=${COMPREFACE_URL:-http://localhost:8080}
# export COMPREFACE_API_KEY=${COMPREFACE_API_KEY:-"00000000-0000-0000-0000-000000000002"}
export COMPREFACE_API_KEY=${COMPREFACE_API_KEY:-"2b6a8351-059a-4ac9-b897-24bc726459e5"}

# override the trained name per folder
export OVERRIDE_TRAINED_NAME=${OVERRIDE_TRAINED_NAME:-"uknown"}

# Check if RUST_LOG is not set
if [ -z "$RUST_LOG" ]; then
    echo "RUST_LOG is not set"
    # Set RUST_LOG to debug for face-recognition-trainer and info for all others
    # this behavior does not include sub modules of the root crate.
    # export RUST_LOG="face-recognition-trainer=debug,info"
    export RUST_LOG="info"
fi

cargo run --bin face-recognition-trainer-cli -- --client-type compreface --client-mode train