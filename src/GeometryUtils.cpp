#define _USE_MATH_DEFINES
#include "GeometryUtils.h"
#include <algorithm>
#include <queue>
#include <cmath>
#include "clipper.hpp"

namespace OCR {

    // Helper functions
    double CrossProduct(const Point2D& a, const Point2D& b, const Point2D& o) {
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    }

    double DistSq(const Point2D& a, const Point2D& b) {
        return (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y);
    }
    
    double Dist(const Point2D& a, const Point2D& b) {
        return std::sqrt(DistSq(a, b));
    }

    double PolygonArea(const std::vector<Point2D>& poly) {
        double area = 0;
        if (poly.size() < 3) return 0;
        for (size_t i = 0; i < poly.size(); ++i) {
            area += CrossProduct(poly[i], poly[(i + 1) % poly.size()], {0,0});
        }
        return std::abs(area) / 2.0;
    }

    double PolygonLength(const std::vector<Point2D>& poly) {
        double len = 0;
        if (poly.size() < 2) return 0;
        for (size_t i = 0; i < poly.size(); ++i) {
            len += Dist(poly[i], poly[(i + 1) % poly.size()]);
        }
        return len;
    }

    std::vector<Quad> GeometryUtils::GetQuadsFromMap(const std::vector<float>& map, int w, int h, float threshold, float unclipRatio) {
        // 1. Threshold
        std::vector<unsigned char> bitMap(w * h, 0);
        for (size_t i = 0; i < map.size(); ++i) {
            if (map[i] > threshold) bitMap[i] = 255;
        }

        // 2. Find Blobs
        std::vector<std::vector<Point2D>> blobs = FindBlobs(bitMap, w, h);

        if (blobs.empty()) return {};

        // 3. Convex Hulls -> Clipper Paths
        ClipperLib::Clipper clipper;
        for (const auto& blob : blobs) {
            std::vector<Point2D> hull = GetConvexHull(blob);
            ClipperLib::Path path;
            for (const auto& p : hull) {
                // Scale up
                path.push_back(ClipperLib::IntPoint((ClipperLib::cInt)(p.x * 1000), (ClipperLib::cInt)(p.y * 1000)));
            }
            clipper.AddPath(path, ClipperLib::ptSubject, true);
        }

        // 4. Union
        ClipperLib::Paths solution;
        clipper.Execute(ClipperLib::ctUnion, solution, ClipperLib::pftNonZero, ClipperLib::pftNonZero);

        // 5. Expand (Unclip) and Convert back
        // 5. Expand (Unclip) and Union (Merge)
        ClipperLib::Clipper unionClipper;

        for (const auto& path : solution) {
            std::vector<Point2D> poly;
            for (const auto& p : path) {
                poly.push_back({ (double)p.X / 1000.0, (double)p.Y / 1000.0 });
            }
            
            if (poly.size() < 3) continue;

            // Calculate offset distance
            double area = PolygonArea(poly);
            double length = PolygonLength(poly);
            if (length <= 0) continue;
            
            double distance = area * unclipRatio / length;
            double offsetDist = distance * 1000.0; // Scale to clipper units

            // Expand individual kernel
            ClipperLib::ClipperOffset offset;
            offset.AddPath(path, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
            
            ClipperLib::Paths expandedPaths;
            offset.Execute(expandedPaths, offsetDist);

            // Add expanded paths to the union clipper
            for(const auto& ep : expandedPaths) {
                 unionClipper.AddPath(ep, ClipperLib::ptSubject, true);
            }
        }

        // Execute Union to merge overlapping expanded regions
        ClipperLib::Paths mergedPaths;
        unionClipper.Execute(ClipperLib::ctUnion, mergedPaths, ClipperLib::pftNonZero, ClipperLib::pftNonZero);

        std::vector<Quad> results;
        for (const auto& path : mergedPaths) {
            std::vector<Point2D> expPoly;
            for (const auto& p : path) {
                expPoly.push_back({ (double)p.X / 1000.0, (double)p.Y / 1000.0 });
            }
            
            if (!expPoly.empty()) {
                std::vector<Point2D> hull = GetConvexHull(expPoly);
                results.push_back(GetMinAreaRect(hull));
            }
        }

        return results;
    }

    std::vector<std::vector<Point2D>> GeometryUtils::FindBlobs(const std::vector<unsigned char>& bitMap, int w, int h) {
        std::vector<std::vector<Point2D>> blobs;
        std::vector<bool> visited(w * h, false);
        
        int dx[] = {1, -1, 0, 0};
        int dy[] = {0, 0, 1, -1};

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int idx = y * w + x;
                if (bitMap[idx] == 255 && !visited[idx]) {
                    // Start BFS
                    std::vector<Point2D> blob;
                    std::queue<int> q;
                    q.push(idx);
                    visited[idx] = true;
                    blob.push_back({(double)x, (double)y});

                    while(!q.empty()){
                        int curr = q.front(); q.pop();
                        int cx = curr % w;
                        int cy = curr / w;

                        for(int i=0; i<4; ++i){
                            int nx = cx + dx[i];
                            int ny = cy + dy[i];
                            if(nx >=0 && nx < w && ny >=0 && ny < h){
                                int nidx = ny * w + nx;
                                if(bitMap[nidx] == 255 && !visited[nidx]){
                                    visited[nidx] = true;
                                    q.push(nidx);
                                    blob.push_back({(double)nx, (double)ny});
                                }
                            }
                        }
                    }
                    if (blob.size() > 10) { 
                        blobs.push_back(blob);
                    }
                }
            }
        }
        return blobs;
    }

    std::vector<Point2D> GeometryUtils::GetConvexHull(const std::vector<Point2D>& points) {
        if (points.size() <= 2) return points;
        
        std::vector<Point2D> P = points;
        std::sort(P.begin(), P.end(), [](const Point2D& a, const Point2D& b) {
            return a.x < b.x || (a.x == b.x && a.y < b.y);
        });

        std::vector<Point2D> H;

        // Lower hull
        for (const auto& p : P) {
            while (H.size() >= 2 && CrossProduct(H[H.size()-2], H.back(), p) <= 0) {
                H.pop_back();
            }
            H.push_back(p);
        }

        // Upper hull
        size_t lower_size = H.size();
        for (int i = P.size() - 2; i >= 0; i--) {
            while (H.size() > lower_size && CrossProduct(H[H.size()-2], H.back(), P[i]) <= 0) {
                H.pop_back();
            }
            H.push_back(P[i]);
        }

        H.pop_back(); 
        return H;
    }

    Quad GeometryUtils::GetMinAreaRect(const std::vector<Point2D>& hull) {
        if (hull.empty()) return {};
        if (hull.size() < 3) {
            Quad q;
            // Simplified fallback
            q.p[0] = hull[0];
            q.p[1] = hull.back();
            q.p[2] = hull[0];
            q.p[3] = hull.back();
            return q;
        }

        double minArea = 1e9; // arb large
        Quad bestQ;
        
        for (size_t i = 0; i < hull.size(); ++i) {
            Point2D p1 = hull[i];
            Point2D p2 = hull[(i + 1) % hull.size()];
            
            double edgeLen = Dist(p1, p2);
            if (edgeLen < 1e-6) continue;

            double dx = (p2.x - p1.x) / edgeLen;
            double dy = (p2.y - p1.y) / edgeLen; 
            
            double nx = -dy;
            double ny = dx;

            double minPara = 1e9, maxPara = -1e9;
            double minPerp = 1e9, maxPerp = -1e9;

            for (const auto& p : hull) {
                double para = p.x * dx + p.y * dy;
                double perp = p.x * nx + p.y * ny;

                if (para < minPara) minPara = para;
                if (para > maxPara) maxPara = para;
                if (perp < minPerp) minPerp = perp;
                if (perp > maxPerp) maxPerp = perp;
            }

            double area = (maxPara - minPara) * (maxPerp - minPerp);
            if (area < minArea) {
                minArea = area;
                
                auto toWorld = [&](double para, double perp) -> Point2D {
                    return { para * dx + perp * nx, para * dy + perp * ny };
                };

                bestQ.p[0] = toWorld(minPara, minPerp);
                bestQ.p[1] = toWorld(maxPara, minPerp);
                bestQ.p[2] = toWorld(maxPara, maxPerp);
                bestQ.p[3] = toWorld(minPara, maxPerp);
            }
        }

        // Reorder points to be: TL, TR, BR, BL (Spatial)
        // 1. Sort by X
        std::vector<Point2D> pts(4);
        for(int i=0; i<4; ++i) pts[i] = bestQ.p[i];
        
        std::sort(pts.begin(), pts.end(), [](const Point2D& a, const Point2D& b) {
            return a.x < b.x;
        });

        // 2. Split Left/Right
        Point2D left[2] = { pts[0], pts[1] };
        Point2D right[2] = { pts[2], pts[3] };

        // 3. Sort by Y
        if (left[0].y > left[1].y) std::swap(left[0], left[1]);
        if (right[0].y > right[1].y) std::swap(right[0], right[1]);

        // 4. Assign: TL, TR, BR, BL
        bestQ.p[0] = left[0];  // TL
        bestQ.p[1] = right[0]; // TR
        bestQ.p[2] = right[1]; // BR
        bestQ.p[3] = left[1];  // BL

        return bestQ;
    }

}
