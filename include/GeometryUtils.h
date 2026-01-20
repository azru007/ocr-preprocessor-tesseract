#pragma once

#include "CoreTypes.h"
#include <vector>

namespace OCR {

    class GeometryUtils {
    public:
        // Thresholds the map, finds blobs, computes hulls, unions them, and returns reduced Quads.
        // map: Flattened float array from inference (N, 1, H, W).
        // w, h: Dimension of the map.
        // threshold: Binary threshold (e.g., 0.3).
        static std::vector<Quad> GetQuadsFromMap(const std::vector<float>& map, int w, int h, float threshold = 0.3f, float unclipRatio = 1.5f);

    private:
        // Helper: Convert boolean map to list of contours/polygons (Vector of Points)
        static std::vector<std::vector<Point2D>> FindBlobs(const std::vector<unsigned char>& bitMap, int w, int h);
        
        // Helper: Compute Convex Hull for a set of points (Monotone Chain)
        static std::vector<Point2D> GetConvexHull(const std::vector<Point2D>& points);
        
        // Helper: Find Minimum Area Rectangle for a sorted convex polygon
        static Quad GetMinAreaRect(const std::vector<Point2D>& hull);
    };

}
