#include "PostProcess.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <vector>
#include <numeric>
#include <cmath>

namespace OCR {

    int PostProcess::GetOtsuThreshold(const std::vector<unsigned char>& grayPixels) {
        if (grayPixels.empty()) return 0;
        
        long histogram[256] = {0};
        for (unsigned char p : grayPixels) histogram[p]++;

        double total = grayPixels.size();
        double currentWeight = 0;
        double currentMean = 0;
        
        // Calculate total mean
        double totalMean = 0;
        for (int i = 0; i < 256; ++i) totalMean += i * histogram[i];
        
        double maxVariance = 0;
        int threshold = 0;

        for (int t = 0; t < 256; ++t) {
            double prob = histogram[t];
            if (prob == 0) continue;

            double weightB = currentWeight + prob;
             // If this becomes total, we are done
            if (weightB == total) break;

            double meanB = (currentMean * currentWeight + t * prob) / weightB;
            double weightF = total - weightB;
            double meanF = (totalMean - meanB * weightB) / weightF;

            double varBetween = weightB * weightF * (meanB - meanF) * (meanB - meanF);
            
            if (varBetween > maxVariance) {
                maxVariance = varBetween;
                threshold = t;
            }

            currentWeight = weightB;
            currentMean = meanB;
        }
        return threshold;
    }

    ImageBuffer PostProcess::Binarize(const ImageBuffer& src) {
        int w = src.w;
        int h = src.h;
        int c = src.channels;

        // 1. Grayscale
        std::vector<unsigned char> gray(w * h);
        if (c >= 3) {
            for (int i = 0; i < w * h; ++i) {
                unsigned char r = src.data[i * c + 0];
                unsigned char g = src.data[i * c + 1];
                unsigned char b = src.data[i * c + 2];
                // Luminance
                gray[i] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
            }
        } else {
            // Already 1 channel
             for (int i = 0; i < w * h; ++i) gray[i] = src.data[i*c];
        }

        // 2. Otsu's Method
        int threshold = GetOtsuThreshold(gray);

        // 3. Apply Threshold
        unsigned char* binData = (unsigned char*)malloc(w * h);
        for (int i = 0; i < w * h; ++i) {
            binData[i] = (gray[i] > threshold) ? 255 : 0;
        }

        return ImageBuffer(binData, w, h, 1, true);
    }

    bool PostProcess::SaveImage(const ImageBuffer& img, const std::string& path) {
        // Automatically detect extension or force png
        return stbi_write_png(path.c_str(), img.w, img.h, img.channels, img.data, img.w * img.channels);
    }

    ImageBuffer PostProcess::AdaptiveBinarize(const ImageBuffer& src, int windowSize, double k, double R) {
        int w = src.w;
        int h = src.h;
        int c = src.channels;

        // 1. Grayscale extraction
        std::vector<unsigned char> gray(w * h);
        if (c >= 3) {
            for (int i = 0; i < w * h; ++i) {
                unsigned char r = src.data[i * c + 0];
                unsigned char g = src.data[i * c + 1];
                unsigned char b = src.data[i * c + 2];
                gray[i] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
            }
        } else {
            for (int i = 0; i < w * h; ++i) gray[i] = src.data[i * c];
        }

        // 2. Integral Images (Sum and Sum of Squares)
        // Using double to prevent overflow
        std::vector<double> integralSum(w * h, 0.0);
        std::vector<double> integralSqSum(w * h, 0.0);

        for (int y = 0; y < h; ++y) {
            double rowSum = 0;
            double rowSqSum = 0;
            for (int x = 0; x < w; ++x) {
                int idx = y * w + x;
                unsigned char val = gray[idx];
                rowSum += val;
                rowSqSum += val * val;

                if (y == 0) {
                    integralSum[idx] = rowSum;
                    integralSqSum[idx] = rowSqSum;
                } else {
                    integralSum[idx] = integralSum[(y - 1) * w + x] + rowSum;
                    integralSqSum[idx] = integralSqSum[(y - 1) * w + x] + rowSqSum;
                }
            }
        }

        // 3. Apply Sauvola Thresholding
        unsigned char* binData = (unsigned char*)malloc(w * h);
        int halfWin = windowSize / 2;

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int x1 = std::max(0, x - halfWin);
                int y1 = std::max(0, y - halfWin);
                int x2 = std::min(w - 1, x + halfWin);
                int y2 = std::min(h - 1, y + halfWin);

                int count = (x2 - x1 + 1) * (y2 - y1 + 1);

                double sum = integralSum[y2 * w + x2];
                double sqSum = integralSqSum[y2 * w + x2];

                if (y1 > 0) {
                    sum -= integralSum[(y1 - 1) * w + x2];
                    sqSum -= integralSqSum[(y1 - 1) * w + x2];
                }
                if (x1 > 0) {
                    sum -= integralSum[y2 * w + (x1 - 1)];
                    sqSum -= integralSqSum[y2 * w + (x1 - 1)];
                }
                if (x1 > 0 && y1 > 0) {
                    sum += integralSum[(y1 - 1) * w + (x1 - 1)];
                    sqSum += integralSqSum[(y1 - 1) * w + (x1 - 1)];
                }

                double mean = sum / count;
                double variance = (sqSum / count) - (mean * mean);
                double stdDev = std::sqrt(std::max(0.0, variance));

                double threshold = mean * (1.0 + k * (stdDev / R - 1.0));

                // Binarize
                // Text is usually darker than background
                // Pixel < Threshold => Text (Black/0)
                // Pixel > Threshold => Background (White/255)
                binData[y * w + x] = (gray[y * w + x] < threshold) ? 0 : 255;
            }
        }
        
        return ImageBuffer(binData, w, h, 1, true);
    }

    ImageBuffer PostProcess::NormalizeBackground(const ImageBuffer& src) {
        int w = src.w;
        int h = src.h;
        int c = src.channels;
        
        if (h < 3) return ImageBuffer(src.data, w, h, c, false); // Too small to process, return shallow copy/original logic handled by caller? 
        // Actually we need to return a new buffer because ImageBuffer owns its data by default in our usage pattern
        // Or we can just copy it.
        
        // Helper to get row average luminance
        auto getRowAvg = [&](int y) -> double {
            double sum = 0;
            for (int x = 0; x < w; ++x) {
                unsigned char grayVal;
                int idx = (y * w + x) * c;
                if (c >= 3) {
                     unsigned char r = src.data[idx];
                     unsigned char g = src.data[idx + 1];
                     unsigned char b = src.data[idx + 2];
                     grayVal = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
                } else {
                     grayVal = src.data[idx];
                }
                sum += grayVal;
            }
            return sum / w;
        };

        // Center Row
        double centerAvg = getRowAvg(h / 2);
        
        // Edge Rows (Top and Bottom)
        double edgeAvg = (getRowAvg(0) + getRowAvg(h - 1)) / 2.0;

        // Logic: 
        // If Center (Text) is Lighter (> value) than Edge (Background), 
        // it means we have Light Text on Dark Background.
        // We want Dark Text on Light Background.
        // So we should INVERT.
        
        bool needInvert = (centerAvg > edgeAvg);
        
        // Create new buffer
        size_t size = (size_t)w * h * c;
        unsigned char* newData = (unsigned char*)malloc(size);
        
        if (needInvert) {
             for (size_t i = 0; i < size; ++i) {
                newData[i] = 255 - src.data[i];
            }
        } else {
             memcpy(newData, src.data, size);
        }

        return ImageBuffer(newData, w, h, c, true);
    }

}
