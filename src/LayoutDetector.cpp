#include "LayoutDetector.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>

// Define STB_IMAGE_RESIZE_IMPLEMENTATION if needed, or implement simple resize
// Since we technically need to include stb_image_resize.h which might not be there...
// We'll implement a simple resize function similar to TextDetector or assume stb_image_resize exists if available.
// TextDetector implemented its own simple bilinear resize. We'll replicate that helper to avoid new dependencies.

namespace OCR {

    // Helper: IoU for NMS
    float ComputeIoU(const LayoutBox& a, const LayoutBox& b) {
        float xA = std::max(a.x1, b.x1);
        float yA = std::max(a.y1, b.y1);
        float xB = std::min(a.x2, b.x2);
        float yB = std::min(a.y2, b.y2);

        float interArea = std::max(0.0f, xB - xA) * std::max(0.0f, yB - yA);
        float boxAArea = (a.x2 - a.x1) * (float)(a.y2 - a.y1);
        float boxBArea = (b.x2 - b.x1) * (float)(b.y2 - b.y1);

        return interArea / (boxAArea + boxBArea - interArea + 1e-5f);
    }

    // Helper: NMS
    std::vector<LayoutBox> NMS(std::vector<LayoutBox>& boxes, float iouThreshold) {
        if (boxes.empty()) return {};

        // Sort by score descending
        std::sort(boxes.begin(), boxes.end(), [](const LayoutBox& a, const LayoutBox& b) {
            return a.score > b.score;
        });

        std::vector<LayoutBox> picked;
        std::vector<bool> suppressed(boxes.size(), false);

        for (size_t i = 0; i < boxes.size(); ++i) {
            if (suppressed[i]) continue;
            picked.push_back(boxes[i]);

            for (size_t j = i + 1; j < boxes.size(); ++j) {
                if (suppressed[j]) continue;
                if (boxes[i].id == boxes[j].id && ComputeIoU(boxes[i], boxes[j]) > iouThreshold) {
                    suppressed[j] = true;
                }
            }
        }
        return picked;
    }

    struct LayoutDetector::Impl {
        Ort::Env env;
        Ort::Session session;
        Ort::MemoryInfo memoryInfo;

        Impl() 
            : env(ORT_LOGGING_LEVEL_WARNING, "Layout_Detector"),
              session(nullptr),
              memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) 
        {}
    };

    LayoutDetector::LayoutDetector() : m_impl(std::make_unique<Impl>()) {}
    LayoutDetector::~LayoutDetector() = default;

    bool LayoutDetector::LoadModel(const std::string& modelPath) {
        try {
            Ort::SessionOptions sessionOptions;
            sessionOptions.SetIntraOpNumThreads(1);
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
            m_impl->session = Ort::Session(m_impl->env, modelPath.c_str(), sessionOptions);
            return true;
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime Error (Layout): " << e.what() << std::endl;
            return false;
        }
    }

    // Reuse resize logic from TextDetector (duplicated for isolation)
    std::vector<float> PreprocessTile(const ImageBuffer& img, int cropX, int cropY, int cropW, int cropH, int targetW, int targetH) {
        std::vector<float> inputTensorValues(targetW * targetH * 3);
        
        // Stats for Layout (Standard ImageNet)
        float mean[] = {0.485f, 0.456f, 0.406f};
        float std[] = {0.229f, 0.224f, 0.225f};

        for (int y = 0; y < targetH; ++y) {
            for (int x = 0; x < targetW; ++x) {
                // Map target (x,y) to source crop coordinates
                float srcX = x * ((float)cropW / targetW) + cropX;
                float srcY = y * ((float)cropH / targetH) + cropY;

                // Nearest Neighbor for speed/simplicity or Bilinear?
                // Let's do simple Nearest Neighbor + boundary Check
                int sx = (int)srcX;
                int sy = (int)srcY;

                // Clamp
                sx = std::min(std::max(sx, 0), img.w - 1);
                sy = std::min(std::max(sy, 0), img.h - 1);

                int pixelIdx = (sy * img.w + sx) * img.channels;

                for (int ch = 0; ch < 3; ++ch) {
                    unsigned char val = img.data[pixelIdx + ch];
                    float normalized = ((val / 255.0f) - mean[ch]) / std[ch];
                    
                    // CHW
                    int tensorIdx = ch * (targetH * targetW) + (y * targetW + x);
                    inputTensorValues[tensorIdx] = normalized;
                }
            }
        }
        return inputTensorValues;
    }

    std::vector<LayoutBox> LayoutDetector::Detect(const ImageBuffer& img) {
        if (!m_impl->session) return {};

        std::vector<LayoutBox> allBoxes;
        
        const int inputSize = 480; // Fixed input size for model
        const int stride = 400;    // Overlap

        int W = img.w;
        int H = img.h;

        // Sliding Window
        for (int y = 0; y < H; y += stride) {
            for (int x = 0; x < W; x += stride) {
                // Define Crop
                // If we are at the edge, we still take 480x480 if possible? 
                // Or just crop what's left and pad? 
                // Better to shift the window back to fit 480x480 if possible, 
                // or resize specific crop. 
                // Guide says "Resize tile to 480x480". 
                // So we take a crop of size min(tile_size, remaining) and RESIZE it to 480x480.
                
                int r = std::min(x + inputSize, W);
                int b = std::min(y + inputSize, H);
                int cropW = r - x;
                int cropH = b - y;
                
                // Preprocess
                std::vector<float> inputTensor = PreprocessTile(img, x, y, cropW, cropH, inputSize, inputSize);

                // Inference
                std::vector<int64_t> inputShape = {1, 3, inputSize, inputSize};
                
                // Inputs
                const char* inputNames[] = {"image", "scale_factor"};
                // scale_factor shape [1, 2]
                // Usually [1.0, 1.0] if we resized to 480x480?
                // Wait, if we cropped a region (cropW x cropH) and resized to (480x480),
                // the effective scale is 480/cropW and 480/cropH.
                // However, some models expect the scale factor to be relative to the original image?
                // The prompt says: "For tiled inference, since we feed 1:1 crops (resized to 480x480), this is usually passed as [1.0, 1.0]"
                float scaleFactor[] = {1.0f, 1.0f}; 
                std::vector<int64_t> scaleShape = {1, 2};

                try {
                     Ort::Value inputImgTensor = Ort::Value::CreateTensor<float>(
                        m_impl->memoryInfo, inputTensor.data(), inputTensor.size(), inputShape.data(), inputShape.size());
                     
                     Ort::Value scaleTensor = Ort::Value::CreateTensor<float>(
                        m_impl->memoryInfo, scaleFactor, 2, scaleShape.data(), scaleShape.size());

                     std::vector<Ort::Value> inputs;
                     inputs.push_back(std::move(inputImgTensor));
                     inputs.push_back(std::move(scaleTensor));

                     const char* outputNames[] = {"fetch_name_0"}; 
                     
                     auto outputs = m_impl->session.Run(
                        Ort::RunOptions{nullptr}, 
                        inputNames, 
                        inputs.data(), 
                        2, 
                        outputNames, 
                        1
                     );
                     
                     // Parse Output: [class_id, score, x1, y1, x2, y2]
                     float* floatArr = outputs[0].GetTensorMutableData<float>();
                     auto info = outputs[0].GetTensorTypeAndShapeInfo();
                     size_t elementCount = info.GetElementCount(); // N * 6
                     
                     // Result might be empty? Check shape
                     // Shape is usually [N, 6]
                     
                     for (size_t k = 0; k < elementCount; k += 6) {
                         int cls = (int)floatArr[k];
                         float score = floatArr[k+1];
                         float dx1 = floatArr[k+2];
                         float dy1 = floatArr[k+3];
                         float dx2 = floatArr[k+4];
                         float dy2 = floatArr[k+5];

                         if (score > 0.5f) { // Threshold
                             // Map back to global coordinates
                             // 1. Un-resize (map 480 -> cropW/H)
                             float localX1 = dx1 * (cropW / (float)inputSize);
                             float localY1 = dy1 * (cropH / (float)inputSize);
                             float localX2 = dx2 * (cropW / (float)inputSize);
                             float localY2 = dy2 * (cropH / (float)inputSize);

                             // 2. Add offset
                             LayoutBox box;
                             box.id = cls;
                             box.score = score;
                             box.x1 = (int)(localX1 + x);
                             box.y1 = (int)(localY1 + y);
                             box.x2 = (int)(localX2 + x);
                             box.y2 = (int)(localY2 + y);
                             
                             // Clamp
                             box.x1 = std::max(0, box.x1);
                             box.y1 = std::max(0, box.y1);
                             box.x2 = std::min(W, box.x2);
                             box.y2 = std::min(H, box.y2);

                             allBoxes.push_back(box);
                         }
                     }

                } catch (const std::exception& e) {
                    std::cerr << "Inference error on tile: " << e.what() << std::endl;
                }
            }
        }

        // Post-Process: NMS
        // Global NMS
        auto finalBoxes = NMS(allBoxes, 0.3f); // 0.3 IoU threshold

        return finalBoxes;
    }

}
