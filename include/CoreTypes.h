#pragma once

#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <cmath>

namespace OCR {

    struct Point2D {
        double x, y;
    };

    struct Rect {
        int x, y, w, h;
    };

    struct Quad {
        Point2D p[4]; // TL, TR, BR, BL order is assumed but not strictly enforced by struct
    };

    // RAII Image Buffer
    struct ImageBuffer {
        unsigned char* data;
        int w;
        int h;
        int channels;
        bool owner; // If true, we delete data on destruction

        ImageBuffer() : data(nullptr), w(0), h(0), channels(0), owner(false) {}

        ImageBuffer(unsigned char* d, int width, int height, int c, bool own = true)
            : data(d), w(width), h(height), channels(c), owner(own) {}

        ~ImageBuffer() {
            if (owner && data) {
                // Determine allocation method if possible, but standard is `stbi_image_free` or `delete[]`.
                // For this project, we might need a custom deleter or strict convention.
                // We'll trust the user of this struct to handle it or use a specific free function.
                // For simplicity in this "raw" struct, we might actually want to avoid automatic delete 
                // unless we know HOW it was allocated. 
                // IMPROVEMENT: Use std::unique_ptr with custom deleter if strictly C++ smart pointers desired.
                // For now, let's keep it simple as requested, but maybe add a helper "free".
                if (data) free(data); // Assuming malloc/stb allocation style for now.
            }
        }

        // Disable copy to prevent double-free
        ImageBuffer(const ImageBuffer&) = delete;
        ImageBuffer& operator=(const ImageBuffer&) = delete;

        // Allow move
        ImageBuffer(ImageBuffer&& other) noexcept 
            : data(other.data), w(other.w), h(other.h), channels(other.channels), owner(other.owner) {
            other.data = nullptr;
            other.owner = false;
        }

        ImageBuffer& operator=(ImageBuffer&& other) noexcept {
            if (this != &other) {
                if (owner && data) free(data);
                data = other.data;
                w = other.w;
                h = other.h;
                channels = other.channels;
                owner = other.owner;
                other.data = nullptr;
                other.owner = false;
            }
            return *this;
        }
    };

}
