#pragma once

#include "CoreTypes.h"
#include <vector>

namespace OCR {

    class ImageWarp {
    public:
        // Warps the region defined by 'quad' in 'src' to a flat rectangle.
        // Returns a new ImageBuffer.
        static ImageBuffer Warp(const ImageBuffer& src, const Quad& quad);

    private:
        // Solves for the 3x3 Homography Matrix mapping srcPoints to dstPoints.
        // Returns 9 elements (row-major).
        static std::vector<double> GetPerspectiveTransform(const Point2D srcP[4], const Point2D dstP[4]);
        
        static void GaussianElimination(double* A, int n);
    };

}
