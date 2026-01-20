cmake_minimum_required(VERSION 3.17)
project(OCRPreprocessor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------
# Includes
# ----------------------------
include(FetchContent)
include_directories(include)
include_directories(extern/stb)

# ----------------------------
# Try ONNX Runtime (Config mode first)
# ----------------------------
find_package(OnnxRuntime QUIET)

# ----------------------------
# If NOT found, handle per-OS
# ----------------------------
if (NOT OnnxRuntime_FOUND)

    message(STATUS "OnnxRuntime not found via Config.")

    if (WIN32)
        message(STATUS "Downloading prebuilt OnnxRuntime for Windows...")

        set(ORT_VERSION "1.17.1")
        set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-win-x64-${ORT_VERSION}.zip")
        set(ORT_DIR_NAME "onnxruntime-win-x64-${ORT_VERSION}")

        FetchContent_Declare(
            OnnxRuntimeBin
            URL ${ORT_URL}
        )
        FetchContent_MakeAvailable(OnnxRuntimeBin)

        set(ONNXRUNTIME_INCLUDE_DIR "${onnxruntimebin_SOURCE_DIR}/include")
        set(ONNXRUNTIME_LIB "${onnxruntimebin_SOURCE_DIR}/lib/onnxruntime.lib")

        file(GLOB ORT_DLLS "${onnxruntimebin_SOURCE_DIR}/lib/*.dll")
        file(COPY ${ORT_DLLS} DESTINATION ${CMAKE_BINARY_DIR})

        if (NOT EXISTS "${ONNXRUNTIME_LIB}")
            message(FATAL_ERROR "Failed to locate onnxruntime.lib in downloaded package.")
        endif()

    elseif (APPLE)
        message(FATAL_ERROR "Automatic download not implemented for macOS. Please install onnxruntime.")

    else()
        # ----------------------------
        # Linux: find manually installed ONNX Runtime
        # ----------------------------
        message(STATUS "Searching for system-installed OnnxRuntime (Linux)...")

        find_library(ONNXRUNTIME_LIB
            NAMES onnxruntime
            PATHS
                /usr/lib
                /usr/local/lib
                /opt/onnxruntime/lib
            REQUIRED
        )

        find_path(ONNXRUNTIME_INCLUDE_DIR
            NAMES onnxruntime_cxx_api.h
            PATHS
                /usr/include
                /usr/local/include
                /opt/onnxruntime/include
            REQUIRED
        )

        message(STATUS "Found ONNX Runtime library: ${ONNXRUNTIME_LIB}")
        message(STATUS "Found ONNX Runtime includes: ${ONNXRUNTIME_INCLUDE_DIR}")
    endif()

endif()

# ----------------------------
# Source Files
# ----------------------------
file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADERS "include/*.h")

add_executable(ocr_preprocessor ${SOURCES} ${HEADERS})

# ----------------------------
# Link ONNX Runtime
# ----------------------------
if (OnnxRuntime_FOUND)
    target_link_libraries(ocr_preprocessor PRIVATE onnxruntime)
else()
    target_include_directories(ocr_preprocessor PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(ocr_preprocessor PRIVATE ${ONNXRUNTIME_LIB})
endif()

# ----------------------------
# Windows defines
# ----------------------------
if (WIN32)
    target_compile_definitions(ocr_preprocessor PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
