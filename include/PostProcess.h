#pragma once

#include "CoreTypes.h"
#include <string>

namespace OCR {

    class PostProcess {
    public:
        // Converts image to grayscale (if needed) and applies Otsu Binarization.
        // Returns a new single-channel ImageBuffer (black/white).
        static ImageBuffer Binarize(const ImageBuffer& src);

        // Saves buffer to file.
        static bool SaveImage(const ImageBuffer& img, const std::string& path);
    };

}
