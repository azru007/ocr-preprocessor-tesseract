#include <iostream>
#include <string>
#include <vector>
#include <fstream>

// Define STB_IMAGE_IMPLEMENTATION here
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "CoreTypes.h"
#include "TextDetector.h"
#include "GeometryUtils.h"
#include "ImageWarp.h"
#include "PostProcess.h"

// Simple file existence check
bool FileExists(const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

// Use C++17 filesystem
#include <filesystem>
namespace fs = std::filesystem;

void PrepareOutputDirectory(const std::string& path) {
    if (fs::exists(path)) {
        fs::remove_all(path); // Clean existing
    }
    fs::create_directories(path);
}


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ocr_preprocessor <image_path> [model_path]" << std::endl;
        return 1;
    }

    std::string imagePath = argv[1];
    std::string modelPath = (argc > 2) ? argv[2] : "det_v5.onnx";
    
#ifdef OCR_OUTPUT_DIR
    std::string outDir = OCR_OUTPUT_DIR;
#else
    std::string outDir = "output";
#endif

    if (!FileExists(imagePath)) {
        std::cerr << "Error: Image not found: " << imagePath << std::endl;
        return 1;
    }
    if (!FileExists(modelPath)) {
        std::cerr << "Error: Model not found: " << modelPath << std::endl;
        return 1;
    }

    // 1. Load Image
    int w, h, c;
    unsigned char* data = stbi_load(imagePath.c_str(), &w, &h, &c, 3); // Force 3 channels
    if (!data) {
        std::cerr << "Failed to load image." << std::endl;
        return 1;
    }
    
    // Wrap in RAII
    OCR::ImageBuffer inputImg(data, w, h, 3, true); // true = own it (will free)
    std::cout << "Loaded image: " << w << "x" << h << std::endl;

    // 2. Initialize Detector
    OCR::TextDetector detector;
    if (!detector.LoadModel(modelPath)) {
        std::cerr << "Failed to load ONNX model." << std::endl;
        return 1;
    }
    std::cout << "Model loaded." << std::endl;

    // 3. Detect
    int mapW, mapH;
    auto map = detector.Detect(inputImg, mapW, mapH);
    if (map.empty()) {
        std::cerr << "Detection failed or empty result." << std::endl;
        return 1;
    }
    std::cout << "Detection Map: " << mapW << "x" << mapH << std::endl;

    // 4. Group Regions (Geometry)    
    float scaleX = (float)mapW / (float)w;
    float scaleY = (float)mapH / (float)h;
    
    // Use Stable Component Analysis
    auto quads = OCR::GeometryUtils::FindStableTextRegions(map, mapW, mapH, 0.3f);
    std::cout << "Found " << quads.size() << " text regions after stable merging." << std::endl;

    // 5. Process Regions
    // Prepare output directory (clean & create)
    PrepareOutputDirectory(outDir);

    int idx = 0;
    for (auto& q : quads) {
        // Scale Quad back to original
        OCR::Quad originalQ;
        for (int i=0; i<4; ++i) {
            originalQ.p[i].x = q.p[i].x / scaleX;
            originalQ.p[i].y = q.p[i].y / scaleY;
        }

        // Warp from Original Image
        OCR::ImageBuffer warped = OCR::ImageWarp::Warp(inputImg, originalQ);
        
        // Binarize
        OCR::ImageBuffer bin = OCR::PostProcess::Binarize(warped);

        // Save
        std::string outName = outDir + "/region_" + std::to_string(idx++) + ".png";
        OCR::PostProcess::SaveImage(bin, outName);
        
        std::cout << "Saved: " << outName << std::endl;
    }

    std::cout << "Done." << std::endl;
    return 0;
}
