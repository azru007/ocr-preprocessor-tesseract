#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm> // for min/max
#include <cmath>     // for floor/ceil
#include <cstring>   // for memset

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

void EnsureOutputDirectory(const std::string& path) {
    if (!fs::exists(path)) {
        fs::create_directories(path);
    }
}

// Helper: Point in Polygon (Ray Raycasting)
bool IsPointInQuad(double x, double y, const OCR::Quad& q) {
    bool inside = false;
    // Quad always has 4 points
    for (int i = 0, j = 3; i < 4; j = i++) {
        if (((q.p[i].y > y) != (q.p[j].y > y)) &&
            (x < (q.p[j].x - q.p[i].x) * (y - q.p[i].y) / (q.p[j].y - q.p[i].y) + q.p[i].x)) {
            inside = !inside;
        }
    }
    return inside;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ocr_preprocessor <image_path_or_dir> [model_path]" << std::endl;
        return 1;
    }

    std::string inputPath = argv[1];
    std::string modelPath = (argc > 2) ? argv[2] : "det_v5.onnx";
    
    // User preference for comparison
    bool runComparison = false;
    std::cout << "Run OCR comparison (Tesseract vs PaddleOCR)? [y/N]: ";
    std::string choice;
    std::getline(std::cin, choice);
    if (choice == "y" || choice == "Y" || choice == "yes") {
        runComparison = true;
    }
    
#ifdef OCR_OUTPUT_DIR
    std::string outDir = OCR_OUTPUT_DIR;
#else
    std::string outDir = "output";
#endif

    if (!fs::exists(inputPath)) {
        std::cerr << "Error: Input path not found: " << inputPath << std::endl;
        return 1;
    }
    if (!FileExists(modelPath)) {
        std::cerr << "Error: Model not found: " << modelPath << std::endl;
        return 1;
    }

    // 2. Initialize Detector
    OCR::TextDetector detector;
    if (!detector.LoadModel(modelPath)) {
        std::cerr << "Failed to load ONNX model." << std::endl;
        return 1;
    }
    std::cout << "Model loaded." << std::endl;

    EnsureOutputDirectory(outDir);

    std::vector<fs::path> images;
    if (fs::is_directory(inputPath)) {
        for (const auto& entry : fs::directory_iterator(inputPath)) {
            auto ext = entry.path().extension().string();
            // Simple extension check
            if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".bmp") {
                images.push_back(entry.path());
            }
        }
    } else {
        images.push_back(inputPath);
    }

    for (const auto& imgPath : images) {
        std::cout << "Processing: " << imgPath.filename().string() << std::endl;
        
        // 1. Load Image
        int w, h, c;
        unsigned char* data = stbi_load(imgPath.string().c_str(), &w, &h, &c, 3);
        if (!data) {
            std::cerr << "Failed to load image: " << imgPath << std::endl;
            continue;
        }
        
        OCR::ImageBuffer inputImg(data, w, h, 3, true);
        
        // 3. Detect
        int mapW, mapH;
        auto map = detector.Detect(inputImg, mapW, mapH);
        if (map.empty()) {
            std::cerr << "Detection failed for: " << imgPath << std::endl;
            continue;
        }

        // 4. Group Regions
        float scaleX = (float)mapW / (float)w;
        float scaleY = (float)mapH / (float)h;
        auto quads = OCR::GeometryUtils::FindStableTextRegions(map, mapW, mapH, 0.3f);

        // Create a white canvas
        size_t imgSize = (size_t)w * h * 3;
        unsigned char* whiteData = (unsigned char*)malloc(imgSize);
        if (!whiteData) continue;
        memset(whiteData, 255, imgSize);
        
        for (const auto& q : quads) {
            OCR::Quad originalQ;
            double centerX = 0, centerY = 0;
            for (int i=0; i<4; ++i) {
                originalQ.p[i].x = q.p[i].x / scaleX;
                originalQ.p[i].y = q.p[i].y / scaleY;
                centerX += originalQ.p[i].x;
                centerY += originalQ.p[i].y;
            }
            centerX /= 4.0;
            centerY /= 4.0;

            OCR::ImageBuffer warped = OCR::ImageWarp::Warp(inputImg, originalQ);
            OCR::ImageBuffer normalized = OCR::PostProcess::NormalizeBackground(warped);
            // Sauvola adaptive binarization: per-pixel threshold from local mean+stddev.
            // Using a larger window (81) for high-res images to cover 1-2 characters.
            // k=0.18 keeps characters thicker for better OCR.
            OCR::ImageBuffer bin = OCR::PostProcess::AdaptiveBinarize(normalized, 81, 0.18, 128.0);

            int pasteX = (int)(originalQ.p[0].x);
            int pasteY = (int)(originalQ.p[0].y);

            for (int y = 0; y < bin.h; ++y) {
                for (int x = 0; x < bin.w; ++x) {
                    int dstX = pasteX + x;
                    int dstY = pasteY + y;
                    if (dstX >= 0 && dstX < w && dstY >= 0 && dstY < h) {
                        if (bin.data[y * bin.w + x] == 0) {
                            int offset = (dstY * w + dstX) * 3;
                            whiteData[offset + 0] = 0;
                            whiteData[offset + 1] = 0;
                            whiteData[offset + 2] = 0;
                        }
                    }
                }
            }
        }

        std::string baseName = imgPath.stem().string();
        std::string outName = outDir + "/" + baseName + "_masked.png";
        OCR::ImageBuffer outputImg(whiteData, w, h, 3, true);
        OCR::PostProcess::SaveImage(outputImg, outName);

        if (runComparison) {
            // Tesseract on Processed
            std::string cmd1 = "tesseract " + outName + " " + outDir + "/" + baseName + "_tesseract_ours -l eng+mal --psm 6 2>/dev/null";
            system(cmd1.c_str());

            // Tesseract on Original
            std::string cmd2 = "tesseract " + imgPath.string() + " " + outDir + "/" + baseName + "_tesseract_original -l eng+mal --psm 3 2>/dev/null";
            system(cmd2.c_str());

            // PaddleOCR on Original
            std::string cmd3 = "export PATH=\"$PATH:/home/linux/.local/bin\" && paddleocr ocr -i " + imgPath.string() + " --use_angle_cls true --lang ml > " + outDir + "/" + baseName + "_paddleocr.txt 2>/dev/null";
            system(cmd3.c_str());
        }
    }
    return 0;
}
