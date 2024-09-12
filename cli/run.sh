#!/bin/bash
# set current directory of the script
# Get the directory path of the script
base_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "base_script_dir: ${base_script_dir}"

# Change the working directory to the script's directory
cd $base_script_dir

# Set default values only if not already set in the environment
# export DATASET_PATH=${DATASET_PATH:-/home/ron/Documents/smart-home/faces-train/lfw_funneled}
export DATASET_PATH=${DATASET_PATH:-/home/ron/Documents/smart-home/faces-train/un-trained}
export OUTPUT_DIR=${OUTPUT_DIR:-./output-errors/}
export DOUBLE_TAKE_URL=${DOUBLE_TAKE_URL:-http://localhost:3000}
export COMPREFACE_URL=${COMPREFACE_URL:-http://10.100.102.5:31844}
# recognition-with-unknown-famous-faces
export COMPREFACE_API_KEY=${COMPREFACE_API_KEY:-"a096fdec-2d71-430c-a50d-c99cf6fe5d49"}

# recognition-with-known-famous-faces
# export COMPREFACE_API_KEY=${COMPREFACE_API_KEY:-"1dfab7a8-0007-4831-adc2-2a1aedb2e3e5"}


# override the trained name per folder
# export OVERRIDE_TRAINED_NAME=${OVERRIDE_TRAINED_NAME:-"uknown"}

# Check if RUST_LOG is not set
if [ -z "$RUST_LOG" ]; then
    echo "RUST_LOG is not set"
    # Set RUST_LOG to debug for face-recognition-trainer and info for all others
    # this behavior does not include sub modules of the root crate.
    # export RUST_LOG="face-recognition-trainer=debug,info"
    export RUST_LOG="info"
fi

# bunyan --color have problem with the progress bar
# cargo run --bin face-recognition-trainer-cli -- --client-type compreface --client-mode recognize | bunyan --color
cargo run --bin face-recognition-trainer-cli -- --client-type compreface --client-mode recognize