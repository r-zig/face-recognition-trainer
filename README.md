# Face Recognition Trainer and Evaluator
This project provides a tool for training and evaluating face recognition systems. It offers the following key features:

## Folder Scanning:
Recursively scans directories, including all sub folders, to gather images for training and evaluation.
-- Labeling: label images based on the name extracted from their parent directory.
## API Integration:
Integrates with multiple face recognition systems via API calls to automate the training process.
## Result Measurement:
Measures the accuracy and effectiveness of the face recognition models after training, providing valuable insights for further tuning.

Whether you're developing a new face recognition model or improving an existing one, this tool streamlines the process by handling the heavy lifting of data preparation, training, and evaluation.

## Usage
To use this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/r-zig/face-recognition-trainer.git

2. Prepare your dataset: Place your face images in a directory, organized into separate folders for each individual. Each folder should be named based on the person you want to train the images for.

3. Verify the target api service address and if need port-forward to allow communication

4. Set the environment variables in the run.sh script:
```bash
export DATASET_PATH=${DATASET_PATH:-/home/faces-train/}
export DOUBLE_TAKE_URL=${DOUBLE_TAKE_URL:-http://localhost:3000}
export COMPREFACE_URL=${COMPREFACE_URL:-http://localhost:8080}
export COMPREFACE_API_KEY=${COMPREFACE_API_KEY:-"00000000-0000-0000-0000-000000000000"}
# Optional, if set - will copy the failure files into this folder
export OUTPUT_DIR=${OUTPUT_DIR:-./output}
5. To train with CompreFace
```bash
cargo run --bin face-recognition-trainer-cli -- --client-type compreface --client-mode train

6. To recognize with CompreFace
```bash
cargo run --bin face-recognition-trainer-cli -- --client-type compreface --client-mode recognize

6. Execute the program using:
```bash
./cli/run.sh

## Contributors
Ron Zigelman ([@r-zig](https://github.com/r-zig))

## License
This project is licensed under the [MIT License](LICENSE).