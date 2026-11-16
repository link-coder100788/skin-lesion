import CoreML
import Foundation

/// A prepared input: S x S RGB in [0, 1], stored channel-first (3 * S * S), as the model expects.
struct PreparedImage {
    let size: Int
    var chw: [Float]

    func pixel(_ x: Int, _ y: Int) -> (Float, Float, Float) {
        let plane = size * size, i = y * size + x
        return (chw[i], chw[plane + i], chw[2 * plane + i])
    }
}

/// FastViT-T12 converted to Core ML: [0, 1] image -> 7 logits, 1024-d embedding, and the
/// Grad-CAM map for p_urgent on the last feature stage (12 x 12, before ReLU).
final class LesionModel: @unchecked Sendable {
    private let model: MLModel
    let size: Int

    init(size: Int) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        model = try MLModel(contentsOf: Resources.url("LesionNet", "mlmodelc"), configuration: cfg)
        self.size = size
    }

    private func input(_ chw: [Float]) throws -> MLFeatureProvider {
        let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: size), NSNumber(value: size)], dataType: .float32)
        chw.withUnsafeBufferPointer { src in
            arr.dataPointer.assumingMemoryBound(to: Float.self).update(from: src.baseAddress!, count: src.count)
        }
        return try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: arr)])
    }

    private static func floats(_ arr: MLMultiArray) -> [Float] {
        (0..<arr.count).map { arr[$0].floatValue }
    }

    func predict(_ image: PreparedImage) throws -> (logits: [Float], embedding: [Float], cam: [Float]) {
        let out = try model.prediction(from: input(image.chw))
        return (Self.floats(out.featureValue(for: "logits")!.multiArrayValue!),
                Self.floats(out.featureValue(for: "embedding")!.multiArrayValue!),
                Self.floats(out.featureValue(for: "cam")!.multiArrayValue!))
    }

    /// Logits for many images at once (used by image SHAP).
    func logits(_ batch: [[Float]]) throws -> [[Float]] {
        let provider = MLArrayBatchProvider(array: try batch.map(input))
        let out = try model.predictions(fromBatch: provider)
        return (0..<out.count).map { Self.floats(out.features(at: $0).featureValue(for: "logits")!.multiArrayValue!) }
    }
}

func softmax(_ logits: [Float], temperature: Float = 1) -> [Float] {
    let z = logits.map { $0 / temperature }
    let m = z.max() ?? 0
    let e = z.map { Foundation.exp($0 - m) }
    let s = e.reduce(0, +)
    return e.map { $0 / s }
}
