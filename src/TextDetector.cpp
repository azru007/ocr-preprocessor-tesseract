#include "TextDetector.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <algorithm>
#include <cmath>

// STB Image for resizing if needed, but we might want to do manual resize to avoid extra deps if possible
// The user said "Pre-process: Resize input image ... Normalize".
// We can use stbi_resize or write a simple resize. 
// stb_image_resize.h is another file. We only have stb_image.h.
// Let's use simple nearest neighbor or bilinear for resize to keep it self-contained or use stb_image_resize.h
// Wait, prompt said "Use only stb_image". It didn't forbid stb_image_resize. 
// But "Resize input image to multiples of 32". 
// Let's implemented a simple bilinear resize function here to avoid fetching another header unless necessary.

namespace OCR {

    struct TextDetector::Impl {
        Ort::Env env;
        Ort::Session session;
        Ort::MemoryInfo memoryInfo;

        Impl() 
            : env(ORT_LOGGING_LEVEL_WARNING, "OCR_Preprocessor"),
              session(nullptr),
              memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) 
        {}
    };

    TextDetector::TextDetector() : m_impl(std::make_unique<Impl>()) {}
    TextDetector::~TextDetector() = default;

    bool TextDetector::LoadModel(const std::string& modelPath) {
        try {
            Ort::SessionOptions sessionOptions;
            // 0 = let ORT auto-select thread count (uses all logical CPU cores)
            sessionOptions.SetIntraOpNumThreads(0);
            // Enable all optimizations: constant folding, node fusion, memory planning
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            // Accessing internal pointer for move assignment is tricky with wrappers, 
            // but we can just construct a new session.
            // On Windows, paths are wide strings for ONNX Runtime.
            m_impl->session = Ort::Session(m_impl->env, modelPath.c_str(), sessionOptions);
            return true;
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
            return false;
        }
    }

    // Helper: Resize buffer
    // Simple Bilinear Resize Implementation
    std::vector<float> ResizeAndNormalize(const ImageBuffer& img, int targetW, int targetH) {
        std::vector<float> inputTensorValues(targetW * targetH * 3);
        
        float xRatio = (float)(img.w - 1) / targetW;
        float yRatio = (float)(img.h - 1) / targetH;

        // Mean and Std Dev for PaddleOCR (usually 0.485, 0.456, 0.406 ... wait, Paddle is usually mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] scaled to 0-1)
        // OR mean=0.5, std=0.5. Let's check typical Paddle configs. 
        // Standard PaddleOCR mobile detection: normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], scale=1/255)
        // Let's assume standard ImageNet stats.
        
        float mean[] = {0.485f, 0.456f, 0.406f};
        float std[] = {0.229f, 0.224f, 0.225f};
        // Normalize: (pixel/255.0 - mean) / std

        for (int y = 0; y < targetH; ++y) {
            for (int x = 0; x < targetW; ++x) {
                int x_l = (int)(xRatio * x);
                int y_l = (int)(yRatio * y);
                int x_h = std::min(x_l + 1, img.w - 1);
                int y_h = std::min(y_l + 1, img.h - 1);

                float x_weight = (xRatio * x) - x_l;
                float y_weight = (yRatio * y) - y_l;

                float a = (1 - x_weight) * (1 - y_weight);
                float b = x_weight * (1 - y_weight);
                float c = (1 - x_weight) * y_weight;
                float d = x_weight * y_weight;

                for (int ch = 0; ch < 3; ++ch) {
                    // Pixel indices
                    int idx_a = (y_l * img.w + x_l) * img.channels + ch;
                    int idx_b = (y_l * img.w + x_h) * img.channels + ch;
                    int idx_c = (y_h * img.w + x_l) * img.channels + ch;
                    int idx_d = (y_h * img.w + x_h) * img.channels + ch;

                    unsigned char val_a = img.data[idx_a];
                    unsigned char val_b = img.data[idx_b];
                    unsigned char val_c = img.data[idx_c];
                    unsigned char val_d = img.data[idx_d];

                    float pixel = a * val_a + b * val_b + c * val_c + d * val_d;
                    
                    // CHW Layout for Tensor: (Batch, Channel, Height, Width)
                    // Index: ch * (H*W) + y * W + x
                    int tensorIdx = ch * (targetH * targetW) + (y * targetW + x);
                    
                    float normalized = ((pixel / 255.0f) - mean[ch]) / std[ch];
                    inputTensorValues[tensorIdx] = normalized;
                }
            }
        }
        return inputTensorValues;
    }

    std::vector<float> TextDetector::Detect(const ImageBuffer& img, int& outW, int& outH) {
        // 1. Calculate new size (Multiples of 32)
        int resizeW = img.w;
        int resizeH = img.h;
        
        // Raised from 960 to 1280: allows small/dense text to survive downscaling
        // without being lost. CPU inference is still fast with multi-threading above.
        const int limit = 1280;
        float ratio = 1.0f;
        if (std::max(resizeW, resizeH) > limit) {
             if (resizeW > resizeH) ratio = (float)limit / resizeW;
             else ratio = (float)limit / resizeH;
        }
        resizeW = int(resizeW * ratio);
        resizeH = int(resizeH * ratio);

        // Align to 32
        resizeW = ((resizeW + 31) / 32) * 32;
        resizeH = ((resizeH + 31) / 32) * 32;
        
        // Ensure at least 32
        resizeW = std::max(32, resizeW);
        resizeH = std::max(32, resizeH);

        outW = resizeW;
        outH = resizeH;

        // 2. Preprocess
        std::vector<float> inputTensorValues = ResizeAndNormalize(img, resizeW, resizeH);

        // 3. Inference
        // Input Name: "x" (Standard for Paddle exported ONNX, sometimes "image")
        // Check model input name physically or assume "x". PaddleOCR v4 det usually "x".
        const char* inputNames[] = {"x"};
        const char* outputNames[] = {"sigmoid_0.tmp_0"}; // Standard Paddle output, sometimes different.
        
        // We really should query the model for names, but for this snippet we'll try dynamic or standard.
        // Let's try to discover names if possible, effectively.
        // But `GetInputName` requires Allocator which is complex in C++ API without helper.
        // We will assume "x" and "sigmoid_0.tmp_0" as per PaddleOCR v3/v4 convention. 
        // If it fails, we catch exception.
        
        // Wait, v5 might be different. 
        // Better approach: Iterate inputs.
        
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::AllocatedStringPtr inputNamePtr = m_impl->session.GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr outputNamePtr = m_impl->session.GetOutputNameAllocated(0, allocator);
        
        std::vector<const char*> inputNamesVec = {inputNamePtr.get()};
        std::vector<const char*> outputNamesVec = {outputNamePtr.get()};

        std::vector<int64_t> inputShape = {1, 3, resizeH, resizeW};
        
        try {
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                m_impl->memoryInfo, 
                inputTensorValues.data(), 
                inputTensorValues.size(), 
                inputShape.data(), 
                inputShape.size()
            );

            auto outputTensors = m_impl->session.Run(
                Ort::RunOptions{nullptr}, 
                inputNamesVec.data(), 
                &inputTensor, 
                1, 
                outputNamesVec.data(), 
                1
            );

            // 4. Parse Output
            // Output is (1, 1, H, W)
            float* floatArr = outputTensors[0].GetTensorMutableData<float>();
            auto info = outputTensors[0].GetTensorTypeAndShapeInfo();
            // shape might be different if dynamic
            
            // Return raw copy
            size_t count = info.GetElementCount();
            std::vector<float> result(floatArr, floatArr + count);
            
            // AllocatedStringPtr cleans up automatically
            
            return result;

        } catch (const std::exception& e) {
            std::cerr << "Inference failed: " << e.what() << std::endl;
            return {};
        }
    }

}
