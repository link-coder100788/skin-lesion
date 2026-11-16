import Foundation

/// The XGBoost fusion model (CNN probabilities + age, sex, site), evaluated tree by tree.
/// Exported by ios/export_assets.py, which checks the same traversal against xgboost.
struct FusionModel: Decodable {
    struct Tree: Decodable {
        let `class`: Int
        let left: [Int]
        let right: [Int]
        let feature: [Int]
        let threshold: [Double]   // split value, or the leaf value at leaves
        let defaultLeft: [Bool]
    }

    let baseScore: [Double]
    let features: [String]
    let trees: [Tree]

    /// Class probabilities for one feature row. NaN = missing (follows the default branch).
    func predict(_ x: [Double]) -> [Double] {
        var margin = baseScore
        for t in trees {
            var n = 0
            while t.left[n] != -1 {
                let v = x[t.feature[n]]
                if v.isNaN {
                    n = t.defaultLeft[n] ? t.left[n] : t.right[n]
                } else {
                    n = v < t.threshold[n] ? t.left[n] : t.right[n]
                }
            }
            margin[t.class] += t.threshold[n]
        }
        let m = margin.max()!
        let e = margin.map { Foundation.exp($0 - m) }
        let s = e.reduce(0, +)
        return e.map { $0 / s }
    }
}

struct PatientInfo: Equatable {
    var age: Int?
    var sex: String?     // "male" / "female" / nil
    var site: String?    // one of PipelineConfig.sites, or nil
}

/// Feature row in the fusion model's column order (src/fusion.py `fusion_features`).
func fusionRow(cnnProbs: [Float], info: PatientInfo, columns: [String]) -> [Double] {
    columns.map { c in
        if c.hasPrefix("cnn_p_") {
            let i = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"].firstIndex(of: String(c.dropFirst(6)))!
            return Double(cnnProbs[i])
        }
        if c == "age" { return info.age.map(Double.init) ?? .nan }
        if c.hasPrefix("sex_") { return info.sex == String(c.dropFirst(4)) ? 1 : 0 }
        if c.hasPrefix("site_") {
            return info.site?.replacingOccurrences(of: " ", with: "_") == String(c.dropFirst(5)) ? 1 : 0
        }
        return .nan
    }
}

/// Exact Shapley values of p_urgent over four feature groups, in probability units
/// (src/explain.py `grouped_urgent_shap`). An absent group takes its values from the
/// background rows, so base + sum(values) == p_urgent(x).
struct MetadataExplanation {
    static let groups = ["image (CNN)", "age", "sex", "body site"]
    let base: Double
    let values: [Double]          // one per group
    let shown: [String]           // what the user entered, e.g. "62", "male", "unknown"
}

struct FusionBackground: Decodable {
    let columns: [String]
    let rows: [[Double?]]

    var matrix: [[Double]] { rows.map { $0.map { $0 ?? .nan } } }
}

func groupOf(_ column: String) -> Int {
    if column.hasPrefix("cnn_p_") { return 0 }
    if column == "age" { return 1 }
    if column.hasPrefix("sex_") { return 2 }
    return 3
}

func explainMetadata(row: [Double], info: PatientInfo, model: FusionModel, background: FusionBackground,
                     urgentIdx: [Int]) -> MetadataExplanation {
    let bg = background.matrix
    let colGroup = background.columns.map(groupOf)
    let k = MetadataExplanation.groups.count
    var value = [Double](repeating: 0, count: 1 << k)
    for coalition in 0..<(1 << k) {
        var total = 0.0
        for b in bg {
            var x = b
            for (j, g) in colGroup.enumerated() where coalition & (1 << g) != 0 { x[j] = row[j] }
            let p = model.predict(x)
            total += urgentIdx.reduce(0) { $0 + p[$1] }
        }
        value[coalition] = total / Double(bg.count)
    }
    func fact(_ n: Int) -> Double { n < 2 ? 1 : Double((2...n).reduce(1, *)) }
    var phi = [Double](repeating: 0, count: k)
    for g in 0..<k {
        for s in 0..<(1 << k) where s & (1 << g) == 0 {
            let r = s.nonzeroBitCount
            let w = fact(r) * fact(k - r - 1) / fact(k)
            phi[g] += w * (value[s | (1 << g)] - value[s])
        }
    }
    let pImg = urgentIdx.reduce(0.0) { $0 + row[$1] }
    let shown = [String(format: "%.2f urgent", pImg),
                 info.age.map(String.init) ?? "unknown",
                 info.sex ?? "unknown",
                 info.site ?? "unknown"]
    return MetadataExplanation(base: value[0], values: phi, shown: shown)
}
