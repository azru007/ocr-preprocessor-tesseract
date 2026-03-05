#include "PostProcess.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <vector>
#include <numeric>
#include <cmath>

namespace OCR {

    // Forward declaration — defined later in this file.
    static void GaussianBlur3x3(std::vector<unsigned char>& gray, int w, int h);

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

        // 1.5 Suppress noise before computing local statistics.
        // A 3x3 Gaussian removes JPEG artifacts and warp-aliasing so that
        // individual noisy pixels don't skew the Sauvola mean/stddev window.
        GaussianBlur3x3(gray, w, h);

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

                // Sauvola's thresholding method
                // R is typically 128 for 8-bit images.
                // If stdDev is very low, it's likely a flat background area.
                // We force it to white (background) to avoid noise specks.
                double threshold;
                if (stdDev < 2.0) {
                    threshold = -1.0;
                } else {
                    threshold = mean * (1.0 + k * (stdDev / R - 1.0));
                }

                // Binarize
                // Pixel < Threshold => Text (Black/0)
                binData[y * w + x] = (gray[y * w + x] < threshold) ? 0 : 255;
            }
        }
        
        return ImageBuffer(binData, w, h, 1, true);
    }

    // Fast 3x3 Gaussian blur (in-place on a gray vector).
    // Kernel: [1 2 1 / 2 4 2 / 1 2 1] / 16
    // Used to suppress noise before thresholding.
    static void GaussianBlur3x3(std::vector<unsigned char>& gray, int w, int h) {
        std::vector<unsigned char> tmp(gray.size());
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                // Clamp-to-edge for border pixels
                auto g = [&](int px, int py) -> int {
                    px = std::max(0, std::min(px, w - 1));
                    py = std::max(0, std::min(py, h - 1));
                    return gray[py * w + px];
                };
                int v = g(x-1,y-1)*1 + g(x,y-1)*2 + g(x+1,y-1)*1
                      + g(x-1,y  )*2 + g(x,y  )*4 + g(x+1,y  )*2
                      + g(x-1,y+1)*1 + g(x,y+1)*2 + g(x+1,y+1)*1;
                tmp[y * w + x] = (unsigned char)(v / 16);
            }
        }
        gray = tmp;
    }


    ImageBuffer PostProcess::NormalizeBackground(const ImageBuffer& src) {
        int w = src.w;
        int h = src.h;
        int c = src.channels;

        // If the image is too small to split into bands, just copy.
        if (h < 5) {
            size_t size = (size_t)w * h * c;
            unsigned char* cp = (unsigned char*)malloc(size);
            memcpy(cp, src.data, size);
            return ImageBuffer(cp, w, h, c, true);
        }

        // Average luminance over a horizontal pixel band [y0, y1).
        // More robust than a single row — a single row can land in blank space.
        auto bandAvg = [&](int y0, int y1) -> double {
            y0 = std::max(0, y0);
            y1 = std::min(h, y1);
            double sum = 0;
            int count = 0;
            for (int y = y0; y < y1; ++y) {
                for (int x = 0; x < w; ++x) {
                    int idx = (y * w + x) * c;
                    double lum;
                    if (c >= 3)
                        lum = 0.299 * src.data[idx] + 0.587 * src.data[idx+1] + 0.114 * src.data[idx+2];
                    else
                        lum = src.data[idx];
                    sum += lum;
                    ++count;
                }
            }
            return count > 0 ? sum / count : 0.0;
        };

        // Compare top+bottom 20% (background border) vs middle 60% (text content).
        int fifth = std::max(1, h / 5);
        double edgeAvg   = (bandAvg(0, fifth) + bandAvg(h - fifth, h)) / 2.0;
        double centerAvg = bandAvg(fifth, h - fifth);

        // If content (center) is brighter than background (border) by more than 5,
        // it is light-on-dark => invert to get dark-on-light for OCR.
        // The +5 dead-zone prevents flipping nearly-uniform patches.
        bool needInvert = (centerAvg > edgeAvg + 5.0);

        size_t size = (size_t)w * h * c;
        unsigned char* newData = (unsigned char*)malloc(size);

        if (needInvert) {
            for (size_t i = 0; i < size; ++i)
                newData[i] = 255 - src.data[i];
        } else {
            memcpy(newData, src.data, size);
        }

        return ImageBuffer(newData, w, h, c, true);
    }

}
