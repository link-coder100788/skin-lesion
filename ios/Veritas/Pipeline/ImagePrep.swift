import CoreGraphics
import UIKit

enum ImagePrep {
    /// Any photo -> S x S RGB in [0, 1], resized full-frame (no crop), like src/triage.py `prepare`.
    /// Drawing through UIGraphicsImageRenderer also applies the photo's EXIF orientation.
    static func prepare(_ image: UIImage, size: Int) -> PreparedImage {
        let fmt = UIGraphicsImageRendererFormat()
        fmt.scale = 1
        fmt.opaque = true
        let target = CGSize(width: size, height: size)
        let resized = UIGraphicsImageRenderer(size: target, format: fmt).image { ctx in
            ctx.cgContext.interpolationQuality = .high
            image.draw(in: CGRect(origin: .zero, size: target))
        }
        return fromCGImage(resized.cgImage!, size: size)
    }

    /// An image that is already S x S (the bundled examples): read its pixels unchanged.
    static func fromCGImage(_ cg: CGImage, size: Int) -> PreparedImage {
        var rgba = [UInt8](repeating: 0, count: size * size * 4)
        let ctx = CGContext(data: &rgba, width: size, height: size, bitsPerComponent: 8, bytesPerRow: size * 4,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: size, height: size))
        let plane = size * size
        var chw = [Float](repeating: 0, count: 3 * plane)
        for i in 0..<plane {
            chw[i] = Float(rgba[4 * i]) / 255
            chw[plane + i] = Float(rgba[4 * i + 1]) / 255
            chw[2 * plane + i] = Float(rgba[4 * i + 2]) / 255
        }
        return PreparedImage(size: size, chw: chw)
    }

    static func toCGImage(_ img: PreparedImage) -> CGImage {
        rgbImage(size: img.size) { x, y in img.pixel(x, y) }
    }

    /// Build an sRGB image from a per-pixel colour function with components in [0, 1].
    static func rgbImage(size: Int, _ color: (Int, Int) -> (Float, Float, Float)) -> CGImage {
        var rgba = [UInt8](repeating: 255, count: size * size * 4)
        for y in 0..<size {
            for x in 0..<size {
                let (r, g, b) = color(x, y)
                let i = 4 * (y * size + x)
                rgba[i] = UInt8(max(0, min(1, r)) * 255 + 0.5)
                rgba[i + 1] = UInt8(max(0, min(1, g)) * 255 + 0.5)
                rgba[i + 2] = UInt8(max(0, min(1, b)) * 255 + 0.5)
            }
        }
        let provider = CGDataProvider(data: Data(rgba) as CFData)!
        return CGImage(width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: size * 4,
                       space: CGColorSpace(name: CGColorSpace.sRGB)!,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)!
    }
}
