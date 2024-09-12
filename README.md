The **Face Recognition Trainer CLI** is a tool for automating the process of training images for facial recognition or verifying the accuracy of face recognition using Compreface or DoubleTake. It scans directories of face images, trains models, or verifies images using the respective APIs, and provides statistics on success and failure rates.

![Alt Text](https://github.com/r-zig/face-recognition-trainer/blob/main/example/screenshot/cli-recongition-example-4.gif)

## Features
- **Train or Recognize Mode**: Train images for facial recognition based on directory names, or verify the recognition of faces in images.
- **Supports Multiple Clients**: Compatible with both Compreface and DoubleTake APIs.
- **Flexible Error Handling**: Choose how to handle images that encounter processing errors (copy, move, or ignore).
- **Automatic Directory Scanning**: Automatically scans directories, including all sub folders and processes images based on the directory name.
- **Labeling**: label images based on the name extracted from their parent directory.
- **Statistics**: Generates statistics for the success and failure of face recognition.

Whether you're developing a new face recognition model or improving an existing one, this tool streamlines the process by handling the heavy lifting of data preparation, training, and evaluation.

## Usage
To use this project, follow these steps:

### Clone the repository:
   ```bash
   git clone https://github.com/r-zig/face-recognition-trainer.git
 ``` 
### Prepare your dataset:
Place your face images in a directory, organized into separate folders for each individual. Each folder should be named based on the person you want to train the images for.

### CLI Arguments
#### --client-type:
Specify the client to use. Options are compreface or doubletake.  
Example: --client-type compreface

#### --client-mode:
Specify whether to run the tool in train or recognize mode.  
Default: train
Example: --client-mode recognize

#### --dataset-path:
The root directory that contains the face images, organized in subdirectories by person name.
Example: --dataset-path ~/datasets/faces

#### --max-request-size:
The maximum size of the request to send to the DoubleTake service. The service is called when the total size of files reaches this limit.  
Default: 10485760 (10MB)  
Example: --max-request-size 5242880  
 (5MB)

## --override-trained-name:
Optionally override the name for all scanned faces, ignoring the folder name.  
Example: --override-trained-name "John_Doe"

#### --output-dir:
Specify a directory to save the results of failed or unrecognized images.  
Example: --output-dir ./output

#### --error-behavior:
Defines how to handle images that encounter errors during processing.
Options:  

##### copy:
The problematic images will be copied to the output directory.
##### move:
The problematic images will be moved to the output directory.
##### ignore:
The problematic images will be ignored and no action will be taken.  
Default: ignore  
Example: --error-behavior move

### CompreFace Setup
To use the CompreFace service, you need to generate an API key:  
Log in to the CompreFace admin interface.  
Create a new Service.  
Copy the generated API Key for that service.  
Set the copied API key as the value for the COMPREFACE_API_KEY environment variable or pass it as an argument in the CLI.  
Example:
   ```bash
   export COMPREFACE_API_KEY="your-copied-api-key"
```

### Examples
#### Training Mode
To train images using the Compreface API:
   ```bash
   cargo run --bin face-recognition-trainer-cli -- --client-type compreface --client-mode train --compreface-api-key 0f3cb33e-fbdf-4fb7-aea5-f293deeb339d --dataset-path ../faces-train/ --compreface-url http://10.100.103.6:31833
   ```
Recognition Mode
To recognize images using the DoubleTake API:
   ```bash
   cargo run --bin face-recognition-trainer-cli -- --client-type doubletake --client-mode recognize
```
Handling Errors by Moving Files
If you want the tool to move problematic images to a specific directory:
   ```bash
   cargo run --bin face-recognition-trainer-cli -- --client-type compreface --client-mode train --error-behavior move --output-dir ./error-images
```

### Installation
To build and run the project:
   ```bash
   cargo build \
   cargo run --bin face-recognition-trainer-cli
```

### Output and Logs
The tool logs its operations and progress. You can control the logging level using the RUST_LOG environment variable.

Example for debugging:
   ```bash
   export RUST_LOG="face-recognition-trainer=debug,info" \
   cargo run --bin face-recognition-trainer-cli
```

## Contributors
Ron Zigelman ([@r-zig](https://github.com/r-zig))

## License
This project is licensed under the [MIT License](LICENSE).
