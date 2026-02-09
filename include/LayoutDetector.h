#pragma once

#include "CoreTypes.h"
#include <string>
#include <vector>
#include <memory>

namespace OCR {

    struct LayoutBox {
        int id;             // Class ID
        float score;        // Confidence
        int x1, y1, x2, y2; // Bounding Box
        std::string label;  // Optional text label
    };

    class LayoutDetector {
    public:
        LayoutDetector();
        ~LayoutDetector();

        bool LoadModel(const std::string& modelPath);
        
        // Detect layout regions using tiled inference
        std::vector<LayoutBox> Detect(const ImageBuffer& img);

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };

}
