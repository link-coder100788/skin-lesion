import XCTest
@testable import Veritas

/// Checks the on-device pipeline against the Python pipeline's outputs (fixtures.json,
/// written by ios/export_assets.py) on the bundled examples.
final class ParityTests: XCTestCase {
    struct Fixture: Decodable {
        let id: String
        let age: Int?
        let sex: String?
        let site: String?
        let cnnProbs: [Float]
        let pUrgent: Float
        let urgent: Bool
        let unlikeTraining: Bool
        let maha: Float
        let knn: Float
        let fusionProbs: [Double]
        let metaShapBase: Double
        let metaShap: [Double]

        var info: PatientInfo { PatientInfo(age: age, sex: sex == "unknown" ? nil : sex, site: site == "unknown" ? nil : site) }
    }

    static let pipeline = try! TriagePipeline()
    let fixtures = Resources.json([Fixture].self, "fixtures")

    func image(_ id: String) -> PreparedImage {
        let ui = UIImage(contentsOfFile: Resources.url(id, "png", subdirectory: "examples").path)!
        return ImagePrep.fromCGImage(ui.cgImage!, size: Self.pipeline.config.imageSize)
    }

    func testCoreMLMatchesPyTorch() throws {
        let p = Self.pipeline
        for f in fixtures {
            let r = try p.run(image(f.id), info: f.info)
            XCTAssertEqual(r.pUrgent, f.pUrgent, accuracy: 0.02, f.id)
            XCTAssertEqual(r.urgent, f.urgent, f.id)
            XCTAssertEqual(r.unlikeTraining, f.unlikeTraining, f.id)
            XCTAssertEqual(r.maha, f.maha, accuracy: 0.03 * f.maha, f.id)
            XCTAssertEqual(r.knn, f.knn, accuracy: 0.01, f.id)
            for (a, b) in zip(r.cnnProbs, f.cnnProbs) { XCTAssertEqual(a, b, accuracy: 0.02, f.id) }
        }
    }

    func testFusionAndMetadataShapMatchPython() {
        let p = Self.pipeline
        for f in fixtures {
            let row = fusionRow(cnnProbs: f.cnnProbs, info: f.info, columns: p.background.columns)
            for (a, b) in zip(p.fusion.predict(row), f.fusionProbs) { XCTAssertEqual(a, b, accuracy: 1e-4, f.id) }
            let m = explainMetadata(row: row, info: f.info, model: p.fusion, background: p.background,
                                    urgentIdx: p.config.urgentIndices)
            XCTAssertEqual(m.base, f.metaShapBase, accuracy: 1e-4)
            for (a, b) in zip(m.values, f.metaShap) { XCTAssertEqual(a, b, accuracy: 1e-4, f.id) }
        }
    }

    /// Owen values are efficient: the map sums to f(image) - f(fully blurred image). The map
    /// should also agree with the Python SHAP map for the same image (2,000 evaluations there).
    func testImageShap() throws {
        let p = Self.pipeline
        let ex = Resources.json([Example].self, "examples", subdirectory: "examples").first { $0.dx == "mel" }!
        let img = image(ex.id)
        let map = try p.imageShap(img)
        let f = { (x: [Float]) in try p.pUrgent(softmax(p.model.logits([x])[0], temperature: p.config.temperature)) }
        let blurred = ImageShap.boxBlur(img, kernel: p.config.shapBlur)
        XCTAssertEqual(map.reduce(0, +), try f(img.chw) - f(blurred.chw), accuracy: 1e-3)

        let ref = ex.map("shap")!
        let s = img.size, g = 16, c = s / g
        func pooled(_ m: [Float]) -> [Float] {
            var out = [Float](repeating: 0, count: g * g)
            for y in 0..<s { for x in 0..<s { out[(y / c) * g + x / c] += m[y * s + x] } }
            return out
        }
        let r = pearson(pooled(map), pooled(ref))
        print("image SHAP vs Python (16x16 pooled) correlation: \(r)")
        XCTAssertGreaterThan(r, 0.5)
    }

    /// The on-device Grad-CAM (hand-written gradient inside the Core ML model) should match
    /// pytorch-grad-cam's map for the same image.
    func testGradCAMMatchesPython() throws {
        let p = Self.pipeline
        for ex in Resources.json([Example].self, "examples", subdirectory: "examples") where ex.hasMaps {
            let r = try p.run(image(ex.id), info: PatientInfo())
            let ref = ex.map("cam")!
            let corr = pearson(r.gradcam, ref)
            print("Grad-CAM vs Python, \(ex.id): \(corr)")
            XCTAssertGreaterThan(corr, 0.98, ex.id)
        }
    }

    func testOODGuardFlagsNonSkinPhoto() throws {
        let r = try Self.pipeline.run(image("cat"), info: PatientInfo())
        XCTAssertTrue(r.unlikeTraining)
    }

    private func pearson(_ a: [Float], _ b: [Float]) -> Float {
        let ma = a.reduce(0, +) / Float(a.count), mb = b.reduce(0, +) / Float(b.count)
        var num: Float = 0, da: Float = 0, db: Float = 0
        for (x, y) in zip(a, b) { num += (x - ma) * (y - mb); da += (x - ma) * (x - ma); db += (y - mb) * (y - mb) }
        return num / (da * db).squareRoot()
    }
}
