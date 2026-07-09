# OCR Preprocessor

A C++ tool designed to detect and extract text regions from images using Deep Learning models (DBNet). The tool handles text detection, perspective warping, and binarization to prepare images for OCR.

## Features

- **Text Detection**: Uses ONNX Runtime with DBNet models.
- **Geometry Processing**: Merges fragmented detections using Stable Component Analysis.
- **Image Warping**: Unwarps perspective-distorted text regions.
- **Binarization**: Converts regions to black and white for better OCR accuracy.
- **Auto-Cleaning**: Automatically clears output directory before saving new results.

## Prerequisites

- **CMake**: Version 3.17 or higher.
- **Compiler**: C++17 compliant compiler (GCC, Clang, or MSVC).
- **Internet**: Required during the first build to download ONNX Runtime dependencies automatically.

## Building the Project

1.  Clone the repository (if applicable).
2.  Create a build directory:
    ```bash
    mkdir build
    cd build
    ```
3.  Configure the project with CMake:
    ```bash
    cmake ..
    ```
    *Note: This step will download the appropriate ONNX Runtime binaries for your platform (Linux/Windows).*
4.  Build the executable:
    ```bash
    make
    ```

## Usage

```bash
./build/ocr_preprocessor <image_path> [model_path]
```

### Arguments

- `<image_path>`: Path to the input image file.
- `[model_path]`: (Optional) Path to the ONNX model file. Defaults to `det_v5.onnx`.

### Important: Model Compatibility

The default model `det_v5.onnx` uses a newer ONNX IR version (v10). The ONNX Runtime version used in this project supports up to IR version 9.

**Please use the provided `det_v5_ir9.onnx` model.**

### Example Command

```bash
./build/ocr_preprocessor test1.png det_v5_ir9.onnx
```

## Output Behavior

The tool is configured to **always store output in the project root's output folder** (`ocp/output/`), regardless of where you run the command from.

**Directory Cleaning**: The `output/` directory is **automatically cleaned** (all files deleted) every time you run the tool to ensure you only see results from the latest run.

