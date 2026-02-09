#define _USE_MATH_DEFINES
#include "ImageWarp.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace OCR {

    ImageBuffer ImageWarp::Warp(const ImageBuffer& src, const Quad& quad) {
        // 1. Determine destination size
        // Use max width/height of the quad sides to minimise aliasing?
        // Or fixed/estimated size.
        double w1 = std::sqrt(pow(quad.p[0].x - quad.p[1].x, 2) + pow(quad.p[0].y - quad.p[1].y, 2));
        double w2 = std::sqrt(pow(quad.p[2].x - quad.p[3].x, 2) + pow(quad.p[2].y - quad.p[3].y, 2));
        double h1 = std::sqrt(pow(quad.p[1].x - quad.p[2].x, 2) + pow(quad.p[1].y - quad.p[2].y, 2));
        double h2 = std::sqrt(pow(quad.p[3].x - quad.p[0].x, 2) + pow(quad.p[3].y - quad.p[0].y, 2));

        int dstW = static_cast<int>(std::max(w1, w2));
        int dstH = static_cast<int>(std::max(h1, h2));
        
        if (dstW <= 0) dstW = 100;
        if (dstH <= 0) dstH = 100;

        // Destination points (Rect)
        Point2D dstP[4] = {
            {0.0, 0.0},
            {(double)dstW, 0.0},
            {(double)dstW, (double)dstH},
            {0.0, (double)dstH}
        };

        // We need Inverse Homography: Dst -> Src
        // So we map dstP -> quad (srcP)
        std::vector<double> H = GetPerspectiveTransform(dstP, quad.p);
        
        // Allocate result
        unsigned char* newData = (unsigned char*)malloc(dstW * dstH * src.channels);
        
        // Bilinear Loop
        for (int y = 0; y < dstH; ++y) {
            for (int x = 0; x < dstW; ++x) {
                // Apply H to (x, y)
                double currX = (double)x;
                double currY = (double)y;
                
                double z = H[6] * currX + H[7] * currY + H[8];
                double srcX = (H[0] * currX + H[1] * currY + H[2]) / z;
                double srcY = (H[3] * currX + H[4] * currY + H[5]) / z;

                // Sample src at (srcX, srcY)
                // Use floor to get the top-left integer coordinate
                int x_l = static_cast<int>(std::floor(srcX));
                int y_l = static_cast<int>(std::floor(srcY));
                int x_h = x_l + 1;
                int y_h = y_l + 1;

                // Weights
                float xw = (float)(srcX - x_l);
                float yw = (float)(srcY - y_l);

                // Boundary checks
                // Helper lambda to get pixel safely
                auto getPixel = [&](int px, int py, int channel) -> unsigned char {
                    // Clamp coordinates to valid range
                    int cx = std::max(0, std::min(px, src.w - 1));
                    int cy = std::max(0, std::min(py, src.h - 1));
                    return src.data[(cy * src.w + cx) * src.channels + channel];
                };

                for (int c = 0; c < src.channels; ++c) {
                    unsigned char val_a = getPixel(x_l, y_l, c);
                    unsigned char val_b = getPixel(x_h, y_l, c);
                    unsigned char val_c = getPixel(x_l, y_h, c);
                    unsigned char val_d = getPixel(x_h, y_h, c);

                    // Interpolate Top and Bottom edges
                    float top = val_a * (1.0f - xw) + val_b * xw;
                    float bot = val_c * (1.0f - xw) + val_d * xw;
                    
                    // Interpolate vertically
                    float finalVal = top * (1.0f - yw) + bot * yw;

                    // Clamp result to 0-255
                    unsigned char res = (unsigned char)std::max(0.0f, std::min(255.0f, finalVal));

                    newData[(y * dstW + x) * src.channels + c] = res;
                }
            }
        }

        return ImageBuffer(newData, dstW, dstH, src.channels, true);
    }

    // Solve H maps src -> dst
    std::vector<double> ImageWarp::GetPerspectiveTransform(const Point2D src[4], const Point2D dst[4]) {
        // 8 Equations for 8 Unknowns (h8 = 1 fixed)
        // x' = (h0 x + h1 y + h2) / (h6 x + h7 y + 1)
        // y' = (h3 x + h4 y + h5) / (h6 x + h7 y + 1)
        //
        // x'(h6 x + h7 y + 1) = h0 x + h1 y + h2
        // h0 x + h1 y + h2 - h6 x x' - h7 y x' = x'
        
        // System A * h = B
        // We will put it in a 8x9 augmented matrix and solve.
        
        double A[8][9];
        memset(A, 0, sizeof(A));

        for (int i = 0; i < 4; ++i) {
            double x = src[i].x;
            double y = src[i].y;
            double u = dst[i].x;
            double v = dst[i].y;

            // Eq 1 for point i (x coordinate)
            // h0 x + h1 y + h2 - h6 x u - h7 y u = u
            A[2*i][0] = x;
            A[2*i][1] = y;
            A[2*i][2] = 1;
            A[2*i][6] = -x * u;
            A[2*i][7] = -y * u;
            A[2*i][8] = u; // RHS

            // Eq 2 for point i (y coordinate)
            // h3 x + h4 y + h5 - h6 x v - h7 y v = v
            A[2*i+1][3] = x;
            A[2*i+1][4] = y;
            A[2*i+1][5] = 1;
            A[2*i+1][6] = -x * v;
            A[2*i+1][7] = -y * v;
            A[2*i+1][8] = v; // RHS
        }

        // Gaussian Elimination
        // Flatten for easier passing if generic, but fixed size here.
        // We need to solve for 8 variables.
        
        for (int i = 0; i < 8; ++i) {
            // Pivot
            int pivot = i;
            for (int k = i + 1; k < 8; ++k) {
                if (std::abs(A[k][i]) > std::abs(A[pivot][i])) pivot = k;
            }
            // Swap rows
            for (int j = i; j < 9; ++j) std::swap(A[i][j], A[pivot][j]);
            
            // Normalize
            double div = A[i][i];
            // If div is close to 0, singular matrix.
            if (std::abs(div) < 1e-8) {
                // Identity fallback?
                return {1,0,0, 0,1,0, 0,0,1};
            }
            
            for (int j = i; j < 9; ++j) A[i][j] /= div;

            // Eliminate
            for (int k = 0; k < 8; ++k) {
                if (k != i) {
                    double factor = A[k][i];
                    for (int j = i; j < 9; ++j) A[k][j] -= factor * A[i][j];
                }
            }
        }

        // Result is in the last column
        std::vector<double> H(9);
        for (int i = 0; i < 8; ++i) H[i] = A[i][8];
        H[8] = 1.0;
        
        return H;
    }

}
