#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm> // for min/max
#include <cmath>     // for floor/ceil

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
    
    // Use standard Unclip and Union
    auto quads = OCR::GeometryUtils::GetQuadsFromMap(map, mapW, mapH, 0.3f, 2.0f);
    std::cout << "Found " << quads.size() << " text regions after merging." << std::endl;

    // 5. Create Masked Output
    // Prepare output directory (clean & create)
    PrepareOutputDirectory(outDir);

    // Create a white canvas
    size_t imgSize = (size_t)w * h * 3;
    unsigned char* whiteData = (unsigned char*)malloc(imgSize);
    if (!whiteData) {
        std::cerr << "Failed to allocate memory for output image." << std::endl;
        return 1;
    }
    memset(whiteData, 255, imgSize); // Fill with white
    
    // Process each quad
    for (const auto& q : quads) {
        // Scale Quad back to original
        OCR::Quad originalQ;
        
        // Compute center for placement
        double centerX = 0, centerY = 0;
        
        for (int i=0; i<4; ++i) {
            originalQ.p[i].x = q.p[i].x / scaleX;
            originalQ.p[i].y = q.p[i].y / scaleY;
            centerX += originalQ.p[i].x;
            centerY += originalQ.p[i].y;
        }
        centerX /= 4.0;
        centerY /= 4.0;

        // 1. Warp (Dewarp/Deskew)
        // This creates a rectilinear image of the text
        OCR::ImageBuffer warped = OCR::ImageWarp::Warp(inputImg, originalQ);
        
        // 2. Binarize (as Mask)
        OCR::ImageBuffer bin = OCR::PostProcess::Binarize(warped);

        // 3. Paste into White Canvas (Centering)
        // Determine top-left position to center the rect on the original quad center
        int pasteX = (int)(centerX - bin.w / 2.0);
        int pasteY = (int)(centerY - bin.h / 2.0);

        for (int y = 0; y < bin.h; ++y) {
            for (int x = 0; x < bin.w; ++x) {
                // Calculate target position
                int dstX = pasteX + x;
                int dstY = pasteY + y;

                // Boundary check
                if (dstX >= 0 && dstX < w && dstY >= 0 && dstY < h) {
                    // Check mask (binarized image)
                    // 0 = Text (Keep), 255 = Background (Clear/Skip)
                    unsigned char maskVal = bin.data[y * bin.w + x];
                    
                    if (maskVal == 0) {
                        // It is text: Copy RGB from original warped image
                        int srcOffset = (y * warped.w + x) * warped.channels;
                        int dstOffset = (dstY * w + dstX) * 3;
                        
                        if (warped.channels == 3) {
                            whiteData[dstOffset]   = warped.data[srcOffset];
                            whiteData[dstOffset+1] = warped.data[srcOffset+1];
                            whiteData[dstOffset+2] = warped.data[srcOffset+2];
                        }
                    }
                    // Else: It is background, leave whiteData as 255 (White)
                }
            }
        }
    }

    // Save the result
    OCR::ImageBuffer outputImg(whiteData, w, h, 3, true); // true = own it
    std::string outName = outDir + "/masked.png";
    if (OCR::PostProcess::SaveImage(outputImg, outName)) {
        std::cout << "Saved masked image to: " << outName << std::endl;
    } else {
        std::cerr << "Failed to save image." << std::endl;
    }

    std::cout << "Done." << std::endl;
    return 0;
}
