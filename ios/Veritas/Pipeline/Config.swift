import Foundation

/// Settings exported from the Python pipeline (ios/export_assets.py -> pipeline.json).
struct PipelineConfig: Decodable {
    let imageSize: Int
    let classes: [String]
    let urgent: [String]
    let sites: [String]
    let temperature: Float
    let tau: Float
    let useFusion: Bool
    let mahaThreshold: Float
    let knnThreshold: Float
    let mspThreshold: Float
    let knnK: Int
    let embeddingDim: Int
    let bankSize: Int
    let shapBlur: Int
    let shapMaxEvals: Int
    let model: String
    let testMetrics: [String: Double]

    var urgentIndices: [Int] { urgent.compactMap { classes.firstIndex(of: $0) } }
}

enum Resources {
    static func url(_ name: String, _ ext: String, subdirectory: String? = nil) -> URL {
        guard let url = Bundle.main.url(forResource: name, withExtension: ext, subdirectory: subdirectory) else {
            fatalError("Missing bundled resource \(name).\(ext). Run ios/export_assets.py first.")
        }
        return url
    }

    static func json<T: Decodable>(_ type: T.Type, _ name: String, subdirectory: String? = nil) -> T {
        do {
            return try JSONDecoder().decode(type, from: Data(contentsOf: url(name, "json", subdirectory: subdirectory)))
        } catch {
            fatalError("Could not decode \(name).json: \(error)")
        }
    }

    /// Little-endian float32 file.
    static func floats(_ name: String) -> [Float] {
        let data = (try? Data(contentsOf: url(name, "bin"))) ?? Data()
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// Little-endian float16 file, widened to Float.
    static func halfs(_ name: String) -> [Float] {
        let data = (try? Data(contentsOf: url(name, "bin"))) ?? Data()
        return data.withUnsafeBytes { $0.bindMemory(to: Float16.self).map(Float.init) }
    }
}

/// The human-readable names used across the app.
enum Lesion {
    static let names: [String: String] = [
        "akiec": "Actinic keratosis / intraepithelial carcinoma",
        "bcc": "Basal cell carcinoma",
        "bkl": "Benign keratosis",
        "df": "Dermatofibroma",
        "mel": "Melanoma",
        "nv": "Nevus (mole)",
        "vasc": "Vascular lesion",
    ]
    static let short: [String: String] = [
        "akiec": "Actinic keratosis", "bcc": "Basal cell carcinoma", "bkl": "Benign keratosis",
        "df": "Dermatofibroma", "mel": "Melanoma", "nv": "Mole", "vasc": "Vascular lesion",
    ]
}
