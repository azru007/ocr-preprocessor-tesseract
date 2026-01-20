#include "PostProcess.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <vector>
#include <numeric>
#include <cmath>

namespace OCR {

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
        // Histogram
        long histogram[256] = {0};
        for (unsigned char p : gray) histogram[p]++;

        double total = w * h;
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

}
