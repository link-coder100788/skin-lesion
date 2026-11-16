import CoreGraphics
import Foundation

/// Heatmap overlays matching src/explain.py `overlay` and `region_name`.
enum Overlay {
    /// Signed maps: red = toward urgent, blue = away (matplotlib "bwr").
    /// Unsigned maps (Grad-CAM): "jet". Both are Gaussian-smoothed (sigma 3) first.
    static func render(_ img: PreparedImage, map: [Float], signed: Bool, alpha: Float = 0.55) -> CGImage {
        let s = img.size
        let sal = gaussian(map, size: s, sigma: 3)
        var weight = [Float](repeating: 0, count: s * s)
        var colors = [(Float, Float, Float)](repeating: (0, 0, 0), count: s * s)
        if signed {
            // robust scale, so a single extreme block doesn't wash out the rest of the map
            let lim = percentile(sal.map(abs), 99.5) + 1e-12
            for i in 0..<(s * s) {
                let t = min(1, max(0, 0.5 + 0.5 * sal[i] / lim))
                colors[i] = bwr(t)
                weight[i] = min(1, abs(sal[i]) / lim)
            }
        } else {
            let lo = sal.min() ?? 0, hi = sal.max() ?? 1
            for i in 0..<(s * s) {
                let t = (sal[i] - lo) / (hi - lo + 1e-12)
                colors[i] = jet(t)
                weight[i] = t
            }
        }
        return ImagePrep.rgbImage(size: s) { x, y in
            let i = y * s + x
            let (r, g, b) = img.pixel(x, y)
            let gray = (r + g + b) / 3 * 0.35
            let a = alpha * min(1, weight[i] * 1.6)
            let c = colors[i]
            return ((gray + r * 0.65) * (1 - a) + c.0 * a,
                    (gray + g * 0.65) * (1 - a) + c.1 * a,
                    (gray + b * 0.65) * (1 - a) + c.2 * a)
        }
    }

    /// Name the 3x3 grid cell holding the most positive saliency ("upper left", "centre", ...).
    static func regionName(_ map: [Float], size s: Int) -> String {
        let c = s / 3
        var cells = [Float](repeating: 0, count: 9)
        for y in 0..<(3 * c) { for x in 0..<(3 * c) { cells[(y / c) * 3 + x / c] += max(0, map[y * s + x]) } }
        let best = cells.indices.max { cells[$0] < cells[$1] }!
        let (r, col) = (best / 3, best % 3)
        if r == 1 && col == 1 { return "centre" }
        let rows = ["upper", "middle", "lower"], cols = ["left", "centre", "right"]
        return "\(rows[r]) \(cols[col])".replacingOccurrences(of: "middle ", with: "")
            .replacingOccurrences(of: " centre", with: "")
    }

    static func percentile(_ v: [Float], _ p: Float) -> Float {
        let sorted = v.sorted()
        let pos = p / 100 * Float(sorted.count - 1)
        let i = Int(pos), f = pos - Float(i)
        return i + 1 < sorted.count ? sorted[i] * (1 - f) + sorted[i + 1] * f : sorted[i]
    }

    static func gaussian(_ map: [Float], size s: Int, sigma: Float) -> [Float] {
        let r = Int((sigma * 4).rounded())      // OpenCV's kernel size for float images: 2*round(4σ)+1
        var k = (-r...r).map { Foundation.exp(-Float($0 * $0) / (2 * sigma * sigma)) }
        let sum = k.reduce(0, +)
        k = k.map { $0 / sum }
        func reflect(_ i: Int) -> Int { i < 0 ? -i : (i >= s ? 2 * (s - 1) - i : i) }
        var tmp = map, out = map
        for y in 0..<s { for x in 0..<s {
            var a: Float = 0
            for j in -r...r { a += k[j + r] * map[y * s + reflect(x + j)] }
            tmp[y * s + x] = a
        } }
        for y in 0..<s { for x in 0..<s {
            var a: Float = 0
            for j in -r...r { a += k[j + r] * tmp[reflect(y + j) * s + x] }
            out[y * s + x] = a
        } }
        return out
    }

    static func bwr(_ t: Float) -> (Float, Float, Float) {
        t < 0.5 ? (2 * t, 2 * t, 1) : (1, 2 - 2 * t, 2 - 2 * t)
    }

    static func jet(_ t: Float) -> (Float, Float, Float) {
        func clamp(_ v: Float) -> Float { min(1, max(0, v)) }
        return (clamp(1.5 - abs(4 * t - 3)), clamp(1.5 - abs(4 * t - 2)), clamp(1.5 - abs(4 * t - 1)))
    }
}
