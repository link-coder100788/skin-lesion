import Foundation

/// Image SHAP on device: Owen values over a recursive split of the image, with a blur masker.
///
/// Mirrors shap's PartitionExplainer with `shap.maskers.Image("blur(k,k)")`, which the Python
/// app uses (src/explain.py `image_shap`): a hidden region shows the box-blurred image, the
/// region tree halves the longer side, and the evaluation budget goes to the nodes whose
/// output changes most. A node that is not split further spreads its credit evenly over
/// its pixels. Positive = pushes p_urgent up ("see a doctor"), negative = pushes it down.
struct ImageShap {
    let image: PreparedImage
    let blurred: PreparedImage
    let f: ([[Float]]) throws -> [Float]   // batch of CHW images -> p_urgent
    var maxEvals = 300
    var batchSize = 20

    struct Rect: Hashable {
        let x0, y0, x1, y1: Int
        var area: Int { (x1 - x0) * (y1 - y0) }
        var children: (Rect, Rect)? {
            let w = x1 - x0, h = y1 - y0
            if w <= 1 && h <= 1 { return nil }
            if w >= h {
                let m = (x0 + x1) / 2
                return (Rect(x0: x0, y0: y0, x1: m, y1: y1), Rect(x0: m, y0: y0, x1: x1, y1: y1))
            }
            let m = (y0 + y1) / 2
            return (Rect(x0: x0, y0: y0, x1: x1, y1: m), Rect(x0: x0, y0: m, x1: x1, y1: y1))
        }
    }

    /// One pending piece of credit: node `rect` switched on in context `on`.
    private struct Item {
        let on: [Rect]       // regions showing the original image (the rest is blurred)
        let f0: Float        // output with `rect` hidden
        let f1: Float        // output with `rect` shown
        let rect: Rect
        let weight: Float
        var priority: Float { abs(f1 - f0) * weight }
    }

    private func compose(_ on: [Rect]) -> [Float] {
        var out = blurred.chw
        let s = image.size, plane = s * s
        for r in on {
            for c in 0..<3 {
                for y in r.y0..<r.y1 {
                    let row = c * plane + y * s
                    out.replaceSubrange((row + r.x0)..<(row + r.x1), with: image.chw[(row + r.x0)..<(row + r.x1)])
                }
            }
        }
        return out
    }

    /// Returns an S x S map (row-major).
    func run(progress: ((Int, Int) -> Void)? = nil) throws -> [Float] {
        let s = image.size
        let root = Rect(x0: 0, y0: 0, x1: s, y1: s)
        let ends = try f([compose([]), compose([root])])
        var evals = 2
        var credit: [Rect: Float] = [:]
        var queue = [Item(on: [], f0: ends[0], f1: ends[1], rect: root, weight: 1)]

        while !queue.isEmpty {
            if evals >= maxEvals { break }
            // expand the highest-priority nodes in one batch
            queue.sort { $0.priority > $1.priority }
            let take = min(batchSize / 2, queue.count, max(1, (maxEvals - evals) / 2))
            let batch = Array(queue.prefix(take))
            queue.removeFirst(take)

            var leaves: [Item] = [], splits: [(Item, Rect, Rect)] = [], inputs: [[Float]] = []
            for item in batch {
                guard let (l, r) = item.rect.children else { leaves.append(item); continue }
                splits.append((item, l, r))
                inputs.append(compose(item.on + [l]))
                inputs.append(compose(item.on + [r]))
            }
            for item in leaves { credit[item.rect, default: 0] += (item.f1 - item.f0) * item.weight }
            let out = inputs.isEmpty ? [] : try f(inputs)
            evals += inputs.count
            progress?(min(evals, maxEvals), maxEvals)

            for (i, (item, l, r)) in splits.enumerated() {
                let f10 = out[2 * i], f01 = out[2 * i + 1]   // only left on, only right on
                let w = item.weight / 2                       // Owen: average over the sibling off / on
                queue.append(Item(on: item.on, f0: item.f0, f1: f10, rect: l, weight: w))
                queue.append(Item(on: item.on + [r], f0: f01, f1: item.f1, rect: l, weight: w))
                queue.append(Item(on: item.on, f0: item.f0, f1: f01, rect: r, weight: w))
                queue.append(Item(on: item.on + [l], f0: f10, f1: item.f1, rect: r, weight: w))
            }
        }
        for item in queue { credit[item.rect, default: 0] += (item.f1 - item.f0) * item.weight }

        var map = [Float](repeating: 0, count: s * s)
        for (r, v) in credit {
            let per = v / Float(r.area)
            for y in r.y0..<r.y1 { for x in r.x0..<r.x1 { map[y * s + x] += per } }
        }
        return map
    }

    /// OpenCV `cv2.blur` (normalised box filter, BORDER_REFLECT_101), per channel.
    static func boxBlur(_ img: PreparedImage, kernel k: Int) -> PreparedImage {
        let s = img.size, plane = s * s
        let lo = k / 2, hi = k - 1 - k / 2      // OpenCV anchors an even kernel at k/2
        func reflect(_ i: Int) -> Int {
            var i = i
            while i < 0 || i >= s { i = i < 0 ? -i : 2 * (s - 1) - i }
            return i
        }
        func blur1D(_ src: [Float], stride: Int, lines: Int, step: Int) -> [Float] {
            var dst = src
            for line in 0..<lines {
                let base = line * stride
                var acc: Float = 0
                for j in -lo...hi { acc += src[base + reflect(j) * step] }
                for i in 0..<s {
                    dst[base + i * step] = acc / Float(k)
                    acc += src[base + reflect(i + hi + 1) * step] - src[base + reflect(i - lo) * step]
                }
            }
            return dst
        }
        var out = img.chw
        for c in 0..<3 {
            var ch = Array(img.chw[(c * plane)..<((c + 1) * plane)])
            ch = blur1D(ch, stride: s, lines: s, step: 1)       // along x
            ch = blur1D(ch, stride: 1, lines: s, step: s)       // along y
            out.replaceSubrange((c * plane)..<((c + 1) * plane), with: ch)
        }
        return PreparedImage(size: s, chw: out)
    }
}
