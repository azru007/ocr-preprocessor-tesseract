#pragma once

#include "CoreTypes.h"
#include <string>

namespace OCR {

    class PostProcess {
    public:
        // Converts image to grayscale (if needed) and applies Otsu Binarization.
        // Returns a new single-channel ImageBuffer (black/white).
        static ImageBuffer Binarize(const ImageBuffer& src);

        // Helper: Compute Otsu threshold from grayscale pixel values
        static int GetOtsuThreshold(const std::vector<unsigned char>& grayPixels);

        // Saves buffer to file.
        static bool SaveImage(const ImageBuffer& img, const std::string& path);

        // Adaptive Thresholding (Sauvola's Method)
        // windowSize: Size of the local neighborhood (e.g., 15)
        // k: Sensitivity factor (0.2 - 0.5 usually)
        // R: Dynamic range of standard deviation (128 for grayscale)
        static ImageBuffer AdaptiveBinarize(const ImageBuffer& src, int windowSize = 25, double k = 0.2, double R = 128.0);
    };

}
