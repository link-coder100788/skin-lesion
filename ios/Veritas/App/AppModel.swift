import SwiftUI
import UIKit

struct Example: Decodable, Identifiable, Hashable {
    let id: String
    let dx: String?
    let why: String
    let age: Int?
    let sex: String
    let site: String
    let hasMaps: Bool

    var title: String { dx.map { Lesion.short[$0] ?? $0 } ?? "Not a skin photo" }

    var image: UIImage { UIImage(contentsOfFile: Resources.url(id, "png", subdirectory: "examples").path)! }

    func map(_ kind: String) -> [Float]? {
        guard hasMaps, let url = Bundle.main.url(forResource: "\(id)_\(kind)", withExtension: "bin",
                                                  subdirectory: "examples"),
              let data = try? Data(contentsOf: url) else { return nil }
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
}

/// What the user has picked: a photo of their own or one of the bundled test-set examples.
enum Selection: Equatable {
    case photo(UIImage)
    case example(Example)

    var image: UIImage {
        switch self {
        case .photo(let img): img
        case .example(let ex): ex.image
        }
    }
}

@Observable
final class Analysis {
    enum Phase { case running, done, failed(String) }

    let selection: Selection
    let info: PatientInfo
    var phase: Phase = .running
    var prepared: PreparedImage?
    var result: TriageResult?
    var summary: String?
    var shapImage: CGImage?
    var camImage: CGImage?
    var shapProgress: Double = 0
    var shapFromCache = false

    init(selection: Selection, info: PatientInfo) {
        self.selection = selection
        self.info = info
    }
}

@MainActor
@Observable
final class AppModel {
    var pipeline: TriagePipeline?
    var loadError: String?
    let examples: [Example]

    init() {
        examples = Resources.json([Example].self, "examples", subdirectory: "examples")
        Task.detached(priority: .userInitiated) {
            do {
                let p = try TriagePipeline()
                await MainActor.run { self.pipeline = p }
            } catch {
                await MainActor.run { self.loadError = error.localizedDescription }
            }
        }
    }

    var config: PipelineConfig? { pipeline?.config }

    func analyze(_ selection: Selection, info: PatientInfo) -> Analysis {
        let analysis = Analysis(selection: selection, info: info)
        guard let pipeline else {
            analysis.phase = .failed("The model is still loading.")
            return analysis
        }
        Task.detached(priority: .userInitiated) {
            await Self.run(pipeline, analysis)
        }
        return analysis
    }

    private nonisolated static func run(_ pipeline: TriagePipeline, _ a: Analysis) async {
        let size = pipeline.config.imageSize
        let image: PreparedImage
        switch a.selection {
        case .photo(let ui): image = ImagePrep.prepare(ui, size: size)
        case .example(let ex): image = ImagePrep.fromCGImage(ex.image.cgImage!, size: size)
        }
        do {
            let r = try pipeline.run(image, info: a.info)
            await MainActor.run {
                a.prepared = image
                a.result = r
                a.phase = .done
            }
            var shap: [Float]
            var cached = false
            if case .example(let ex) = a.selection, let m = ex.map("shap") {
                shap = m
                cached = true
                if let cam = ex.map("cam") {
                    let camImage = Overlay.render(image, map: cam, signed: false)
                    await MainActor.run { a.camImage = camImage }
                }
            } else {
                shap = try pipeline.imageShap(image) { done, total in
                    Task { @MainActor in a.shapProgress = Double(done) / Double(total) }
                }
            }
            let overlay = Overlay.render(image, map: shap, signed: true)
            let text = pipeline.summary(r, region: Overlay.regionName(shap, size: size))
            let fromCache = cached
            await MainActor.run {
                a.shapImage = overlay
                a.shapFromCache = fromCache
                a.shapProgress = 1
                a.summary = text
            }
        } catch {
            await MainActor.run { a.phase = .failed(error.localizedDescription) }
        }
    }
}
