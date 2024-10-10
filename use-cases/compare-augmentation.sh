#!/bin/bash
# set current directory of the script
# Get the directory path of the script
base_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "base_script_dir: ${base_script_dir}"

# Change the working directory to the script's directory
cd $base_script_dir

# Load variables from .env
if [ -f .env ]; then
  source .env
else
  echo ".env file not found!"
  exit 1
fi

# we should take 2 compreface keys:
# one for recognition-with-unknown-famous-faces and the other for
# it will called RECOGNITION_WITH_UNKNOWN_FAMOUS_FACES_KEY
# recognition-with-unknown-famous-faces-augmentation-on-knowns.
# it will called RECOGNITION_WITH_UNKNOWN_FAMOUS_FACES_AUGMENTATION_ON_KNOWNS_KEY
# TRAINED_KNOWN_FACES_FOLDER set folder of our known trained faces to be the same for both cases.
# TRAINED_UNKNOWN_FACES_FOLDER set folder of our unknown trained faces to be the same for both cases.
# AUGMENTED_KNOWN_FACES_FOLDER set folder of our known faces after augmentation.
# TEST_FOLDER used for the recognition process.

# take the known faces and augment them in a way that create the same folder tree (augment/{person name}/name_augment_{1..N}.jpg).
# python3 ../augmentation/augmentation.py $TRAINED_KNOWN_FACES_FOLDER $AUGMENTED_KNOWN_FACES_FOLDER 10

if [ -z "$RUST_LOG" ]; then
    echo "RUST_LOG is not set"
    export RUST_LOG="info"
fi
# train the model with the normal known faces for the key recognition-with-unknown-famous-faces-augmentation-on-knowns.
while true; do
    read -p "Do you want to train the model with the normal KNOWN faces? (y/N): " is_train_known
    case $is_train_known in
        [Yy]* ) is_train_known=true; break;;
        [Nn]* ) is_train_known=false; break;;
        "" ) is_train_known=false; break;;
        * ) echo "Please answer Yes (Y) or No (N). (No by default)";;
    esac
done

cd ../cli
if $is_train_known
then
    cargo run --bin face-recognition-trainer-cli -- \
    --client-type compreface \
    --client-mode train \
    --error-behavior move \
    --compreface-api-key $RECOGNITION_WITH_UNKNOWN_FAMOUS_FACES_AUGMENTATION_ON_KNOWNS_KEY \
    --dataset-path $TRAINED_KNOWN_FACES_FOLDER
    
fi

# train the model with the augmented known faces for the key recognition-with-unknown-famous-faces-augmentation-on-knowns.
while true; do
    read -p "Do you want to train the model with the augmented KNOWN faces? (y/N): " is_train_known_augmented
    case $is_train_known_augmented in
        [Yy]* ) is_train_known_augmented=true; break;;
        [Nn]* ) is_train_known_augmented=false; break;;
        "" ) is_train_known_augmented=false; break;;
        * ) echo "Please answer Yes (Y) or No (N). (No by default)";;
    esac
done

if $is_train_known_augmented
then
    cargo run --bin face-recognition-trainer-cli -- \
    --client-type compreface \
    --client-mode train \
    --error-behavior move \
    --compreface-api-key $RECOGNITION_WITH_UNKNOWN_FAMOUS_FACES_AUGMENTATION_ON_KNOWNS_KEY \
    --dataset-path $AUGMENTED_KNOWN_FACES_FOLDER
fi

# ask the user if we should train the model with the normal unknown faces for the key recognition-with-unknown-famous-faces-augmentation-on-knowns.
while true; do
    read -p "Do you want to train the model with the unknown famous faces for the key recognition-with-unknown-famous-faces-augmentation-on-knowns? (y/N): " is_train_famouse
    case $is_train_famouse in
        [Yy]* ) is_train_famouse=true; break;;
        [Nn]* ) is_train_famouse=false; break;;
        "" ) is_train_famouse=false; break;;
        * ) echo "Please answer Yes (Y) or No (N). (No by default)";;
    esac
done

if $is_train_famouse
then
    # train the model with the famouse faces for the key recognition-with-unknown-famous-faces-augmentation-on-knowns.
    cargo run --bin face-recognition-trainer-cli -- \
    --client-type compreface \
    --client-mode train \
    --error-behavior move \
    --compreface-api-key $RECOGNITION_WITH_UNKNOWN_FAMOUS_FACES_AUGMENTATION_ON_KNOWNS_KEY \
    --dataset-path $TRAINED_UNKNOWN_FACES_FOLDER \
    --override-trained-name "unknown"
fi

# wait for the training to finish. (how we can know that the process finished? we can check the cpu usage and when it is low we can assume that the process finished).
# Wait for the training processes to finish by checking CPU usage
while $is_train_famouse || $is_train_known_augmented || $is_train_known; do
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    if (( $(echo "$cpu_usage < 10.0" | bc -l) )); then
        echo "CPU usage is low, assuming training processes have finished."
        break
    fi
    echo "Waiting for training processes to finish..."
    sleep 10
done

# run the recognition process with the key recognition-with-unknown-famous-faces-augmentation-on-knowns. and store the results.
cargo run --bin face-recognition-trainer-cli -- \
    --client-type compreface \
    --client-mode recognize \
    --error-behavior ignore \
    --compreface-api-key $RECOGNITION_WITH_UNKNOWN_FAMOUS_FACES_AUGMENTATION_ON_KNOWNS_KEY \
    --dataset-path $TEST_FOLDER

while true; do
    read -p "Continue to the next test? (y/N): " is_continue
      case $is_continue in
        [Yy]* ) is_continue=true; break;;
        [Nn]* ) is_continue=false; break;;
        "" ) is_continue=false; break;;
        * ) echo "Please answer Yes (Y) or No (N). (No by default)";;
    esac
done
if $is_continue
then
    # run the recognition process with the key recognition-with-unknown-famous-faces. and store the results.
    cargo run --bin face-recognition-trainer-cli -- \
        --client-type compreface \
        --client-mode recognize \
        --error-behavior ignore \
        --compreface-api-key $RECOGNITION_WITH_UNKNOWN_FAMOUS_FACES_KEY \
        --dataset-path $TEST_FOLDER
fi
# compare the results of the two runs and see if the augmentation helped or not.