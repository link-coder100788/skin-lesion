import Accelerate
import Foundation

/// Flags inputs unlike the training data (src/triage.py `OODDetector`). An image is
/// "unlike training" if its kNN cosine distance (k-th neighbour) to the training embeddings
/// or its Mahalanobis distance to the nearest class mean exceeds the validation 99th percentile.
final class OODGuard: @unchecked Sendable {
    private let means: [Float]       // classes x d
    private let precision: [Float]   // d x d
    private let bank: [Float]        // n x d, L2-normalised
    private let d: Int, n: Int, k: Int
    private let mahaThreshold: Float, knnThreshold: Float

    init(config: PipelineConfig) {
        means = Resources.floats("ood_means")
        precision = Resources.floats("ood_precision")
        bank = Resources.halfs("ood_bank")
        d = config.embeddingDim
        n = config.bankSize
        k = config.knnK
        mahaThreshold = config.mahaThreshold
        knnThreshold = config.knnThreshold
    }

    func mahalanobis(_ emb: [Float]) -> Float {
        var best = Float.infinity
        var pd = [Float](repeating: 0, count: d)
        for c in 0..<(means.count / d) {
            let diff = (0..<d).map { emb[$0] - means[c * d + $0] }
            cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(d), Int32(d), 1, precision, Int32(d), diff, 1, 0, &pd, 1)
            best = min(best, vDSP.dot(diff, pd))
        }
        return best
    }

    func knnDistance(_ emb: [Float]) -> Float {
        let norm = sqrt(vDSP.sumOfSquares(emb))
        let z = emb.map { $0 / norm }
        var sims = [Float](repeating: 0, count: n)
        cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(n), Int32(d), 1, bank, Int32(d), z, 1, 0, &sims, 1)
        sims.sort(by: >)
        return 1 - sims[k - 1]
    }

    func unlikeTraining(_ emb: [Float]) -> (flag: Bool, maha: Float, knn: Float) {
        let m = mahalanobis(emb), kd = knnDistance(emb)
        return (m > mahaThreshold || kd > knnThreshold, m, kd)
    }
}
