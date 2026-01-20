@echo off
echo ==========================================
echo OCR Preprocessor Build Script
echo ==========================================

:: Check for CMake
where cmake >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] CMake is not found in your PATH.
    echo Please install CMake and add it to your PATH.
    echo Download: https://cmake.org/download/
    pause
    exit /b 1
)

:: Check for C++ Compiler
where g++ >nul 2>nul
if %errorlevel% == 0 (
    echo [INFO] Found MinGW/GCC.
    set GENERATOR="MinGW Makefiles"
) else (
    echo [INFO] g++ not found, checking for MSVC...
    where cl >nul 2>nul
    if %errorlevel% == 0 (
        echo [INFO] Found MSVC.
        set GENERATOR="Visual Studio 16 2019"
        :: Adjust generator as needed or let CMake Decide
        set GENERATOR=""
    ) else (
        echo [WARNING] No standard C++ compiler found (g++ or cl). CMake might fail if no compiler is detected.
    )
)

if not exist build mkdir build
cd build

echo [INFO] Running CMake...
if defined GENERATOR (
     if "%GENERATOR%"=="" (
         cmake ..
     ) else (
         cmake -G %GENERATOR% ..
     )
) else (
    cmake ..
)

if %errorlevel% neq 0 (
    echo [ERROR] CMake Configuration failed.
    echo Make sure you have 'onnxruntime' installed or configured.
    echo You may need to set -DONNXRUNTIME_ROOTDIR=path/to/onnxruntime
    pause
    cd ..
    exit /b 1
)

echo [INFO] Building...
cmake --build . --config Release

if %errorlevel% neq 0 (
    echo [ERROR] Build failed.
    pause
    cd ..
    exit /b 1
)

echo [SUCCESS] Build completed. Executable should be in build/Release or build/
cd ..
pause
