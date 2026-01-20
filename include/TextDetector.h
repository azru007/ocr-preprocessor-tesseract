#pragma once

#include "CoreTypes.h"
#include <string>
#include <vector>
#include <memory>

// Forward declaration to avoid exposing ONNX headers in public interface if possible, 
// but for simplicity we might include them data-hiding styles or PIMPL.
// Here we will just include the header in the cpp or use a void* for the session if we want to isolate.
// Let's keep it simple.

namespace OCR {

    class TextDetector {
    public:
        TextDetector();
        ~TextDetector();

        // Load the ONNX model
        bool LoadModel(const std::string& modelPath);

        // Run inference
        // Returns a float map (probability heatmap) normalized 0-1
        // outputWidth/Height might differ from input if resized.
        std::vector<float> Detect(const ImageBuffer& img, int& outW, int& outH);

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };

}
