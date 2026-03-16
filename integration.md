# DBNet Android Integration Guide

This document provides a comprehensive, structured guide describing how the C++ DBNet OCR Preprocessor model was integrated into the existing Java-based Android OCR Application.

---

## 1. Architecture Overview

The core Android application is built using Java and utilizes Tesseract for text recognition (OCR). To enhance recognition accuracy, the **DBNet preprocessor** (written natively in C++) is integrated into the pipeline to clean, skew-correct, and binarize images *before* they are sent to the OCR engine.

To bridge the gap between Java and C++, the integration relies on three core technologies:
*   **JNI (Java Native Interface):** Facilitates communication between Android (Java) and the C++ functions.
*   **Android NDK (Native Development Kit):** Compiles the C++ source code into an Android-compatible shared library (`.so`).
*   **Microsoft ONNX Runtime:** Executes the neural network model (`det_v5_ir9.onnx`) directly within the C++ environment natively on the device.

---

## 2. Integration File Structure

The project directory was updated to house the native components seamlessly within the standard Android structure:

```text
app/src/main/
├── assets/
│   └── det_v5_ir9.onnx             # The DBNet neural network model
├── cpp/
│   ├── CMakeLists.txt              # The Build script for compiling C++ into .so
│   ├── native-lib.cpp              # The JNI bridge connecting Java -> C++
│   ├── PostProcess.cpp / .h        # C++ routines for image binarization & logic
│   ├── TextDetector.cpp / .h       # C++ class to evaluate the ONNX model
│   ├── GeometryUtils.cpp / .h      # C++ utilities for clipping and polygons
│   └── CoreTypes.h                 # Core C++ structs and data types
└── java/com/project/TextGrab/
    ├── MainActivity.java           # Triggers the preprocessing pipeline
    ├── utils/Utils.java            # Helper method to route images based on settings
    └── ocr/DBNetPreprocessor.java  # The Android/Java wrapper over the JNI methods
```

---

## 3. Build System Configuration

Android Studio uses **Gradle** and **CMake** to compile the C++ source code natively.

### 3.1. CMake Configurations (`app/src/main/cpp/CMakeLists.txt`)
The build script performs the following tasks:
1.  **Fetches Dependencies:** Downloads the ONNX Runtime Android package (`.aar`) directly from Maven to expose the necessary C++ headers (`include_directories`).
2.  **Defines Target Library:** Groups all local C++ source files (`*.cpp`) into a single output library called `ocr_preprocessor`.
3.  **Links NDK Libraries:** Statically links standard Android libraries such as `log` (for `__android_log_print`) and `jnigraphics` (for handling Android Bitmaps natively) along with `onnxruntime`.

### 3.2. Gradle Configurations (`app/build.gradle`)
The application-level `build.gradle` was modified to tell Android Studio to compile the C++ code:
```gradle
android {
    ...
    externalNativeBuild {
        cmake {
            path "src/main/cpp/CMakeLists.txt"
        }
    }
}

dependencies {
    // Adds ONNX Runtime capabilities to the APK
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.17.1'
}
```

---

## 4. The JNI Bridge (`native-lib.cpp`)

The JNI layer acts as the physical translator between Java parameters and C++ memory. 

The `native-lib.cpp` file surfaces the function `Java_com_project_TextGrab_ocr_DBNetPreprocessor_processImageNative`, which executes the following routine:
1.  **Extracts Pixels:** Uses `#include <android/bitmap.h>` to lock the Android `Bitmap` object in memory, gaining direct access to its raw `RGBA` pixel bytes (`unsigned char*`).
2.  **Model Execution:** Instantiates `OCR::TextDetector` using the local filepath of the ONNX model.
3.  **Post-Processing Pipeline:** Executes `OCR::PostProcess` routines like adaptive binarization, background normalization, and cropping on the raw pixel data.
4.  **Java Return:** Injects the fully cleaned and processed C++ memory buffer back into a new Android `Bitmap` object and unlocks the memory block, returning the `Bitmap` explicitly to the Java caller.

---

## 5. The Java Interface (`DBNetPreprocessor.java`)

Before C++ can act on files, the Java system must prepare them. `DBNetPreprocessor.java` handles Android-specific limitations:
*   **Asset Extraction:** Standard C++ standard libraries (`std::ifstream`) cannot read files directly out of a compressed Android APK's `/assets/` directory. Therefore, this Java class streams the `det_v5_ir9.onnx` file to the device's internal storage (`context.getFilesDir()`) upon first launch.
*   **JNI Invocation:** It serves as the primary gateway, passing the Android `Bitmap` and the absolute physical path of the ONNX model to the native C++ method.

---

## 6. Execution Flow and UI Hooks

The DBNet preprocessor is conditionally activated by an existing app setting.

### 6.1. UI Interception
In `MainActivity.java`, during the background OCR thread (`ConvertImageToText`), the system checks the user's preference using `Utils.isPreProcessImage()`.
*   **Toggle ON:** The app calls `Utils.preProcessBitmap(context, bitmap)`, passing execution to `DBNetPreprocessor.process()`, which executes the C++ DBNet pipeline natively.
*   **Toggle OFF:** The app falls back to its legacy, lightweight Java-based preprocessing methods.

### 6.2. UI Feedback State
Because the DBNet ONNX model inference is resource-intensive, the loading dialogue dynamically alerts the user when it is actively running the C++ code:
*   Updating the Progress Tracker text to display: `"propossing model tunnong"`
*   Updating the standard text once complete to: `"Recognizing Text..."` (Hand-off to Tesseract).

---

## 7. Execution Summary List

When the user initiates an OCR request with processing enabled, the data flows as follows:

1.  **Java:** Extracts an image (`Bitmap`) and user triggers the OCR scan.
2.  **Java:** Dispalys the loading screen evaluating conditions (`isPreProcessImage`).
3.  **Java:** Calls the Native Interface (`DBNetPreprocessor.processImageNative()`).
4.  **C++:** Locks the memory, converts the Android `Bitmap` into a standard C++ byte array.
5.  **C++:** Executes the text discovery neural network (`ONNX Runtime`).
6.  **C++:** Binarizes and cleans the discovered rectangular areas, returning a new `Bitmap`.
7.  **Java:** Hands the pristine, processed `Bitmap` over to the `ImageTextReader` (Tesseract) engine.
8.  **Java:** Renders the final textual string result on the user's screen.