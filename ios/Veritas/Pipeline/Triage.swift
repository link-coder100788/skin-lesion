import Foundation

enum TriageText {
    static let disclaimer = "Student research demo, not a medical device and not a diagnosis. "
        + "Photos are analysed on this iPhone and never stored or uploaded."
    static let urgentAdvice = "Features of this spot resemble skin cancers in the training data. "
        + "Please book an appointment with a doctor or dermatologist."
    static let benignAdvice = "If it changes in size, shape, or color, see a doctor."
    static let unlikeAdvice = "This photo looks unlike the dermoscopy images the model was trained on, "
        + "so it cannot give a verdict. If you are worried about a spot, see a doctor."
}

struct TriageResult {
    let cnnProbs: [Float]          // calibrated CNN probabilities
    let probs: [Float]             // probabilities the decision uses
    let pUrgent: Float
    let tau: Float
    let urgent: Bool
    let unlikeTraining: Bool
    let lowConfidence: Bool
    let maha: Float
    let knn: Float
    let fusionRow: [Double]
    let metadata: MetadataExplanation
}

/// On-device version of src/triage.py `TriagePipeline`: CNN -> calibration -> triage level,
/// OOD guard, metadata explanation. Everything stays on the phone.
final class TriagePipeline: @unchecked Sendable {
    let config: PipelineConfig
    let model: LesionModel
    let ood: OODGuard
    let fusion: FusionModel
    let background: FusionBackground

    init() throws {
        config = Resources.json(PipelineConfig.self, "pipeline")
        model = try LesionModel(size: config.imageSize)
        ood = OODGuard(config: config)
        fusion = Resources.json(FusionModel.self, "xgb")
        background = Resources.json(FusionBackground.self, "background")
    }

    func pUrgent(_ probs: [Float]) -> Float { config.urgentIndices.reduce(0) { $0 + probs[$1] } }
    func pUrgent(_ probs: [Double]) -> Double { config.urgentIndices.reduce(0) { $0 + probs[$1] } }

    func run(_ image: PreparedImage, info: PatientInfo) throws -> TriageResult {
        let (logits, emb) = try model.predict(image)
        let cnn = softmax(logits, temperature: config.temperature)
        let row = fusionRow(cnnProbs: cnn, info: info, columns: background.columns)
        let probs = config.useFusion ? fusion.predict(row).map(Float.init) : cnn
        let p = pUrgent(probs)
        let flags = ood.unlikeTraining(emb)
        let meta = explainMetadata(row: row, info: info, model: fusion, background: background,
                                   urgentIdx: config.urgentIndices)
        return TriageResult(cnnProbs: cnn, probs: probs, pUrgent: p, tau: config.tau, urgent: p >= config.tau,
                            unlikeTraining: flags.flag, lowConfidence: (cnn.max() ?? 0) < config.mspThreshold,
                            maha: flags.maha, knn: flags.knn, fusionRow: row, metadata: meta)
    }

    func imageShap(_ image: PreparedImage, progress: ((Int, Int) -> Void)? = nil) throws -> [Float] {
        let blurred = ImageShap.boxBlur(image, kernel: config.shapBlur)
        let shap = ImageShap(image: image, blurred: blurred, f: { [self] batch in
            try model.logits(batch).map { pUrgent(softmax($0, temperature: config.temperature)) }
        }, maxEvals: config.shapMaxEvals)
        return try shap.run(progress: progress)
    }

    /// Deterministic plain-English summary (src/triage.py `summary_text`).
    func summary(_ r: TriageResult, region: String) -> String {
        if r.unlikeTraining {
            return "This image looks unlike the dermoscopy photos the model was trained on (for example a "
                + "phone photo, a non-skin image, or a very different camera), so the model cannot assess it. "
                + "The heatmap is shown for transparency only."
        }
        let toward = r.urgent ? "see a doctor" : "lower concern"
        var parts = [String(format: "The model's urgent score is %.2f, %@ the decision threshold of %.2f, "
                                + "so it leans toward \u{201C}%@\u{201D}.",
                            r.pUrgent, r.urgent ? "above" : "below", r.tau, toward)]
        parts.append("The image regions that pushed the score up most were in the \(region) of the photo.")
        let m = r.metadata
        let given = (1..<MetadataExplanation.groups.count).filter { m.shown[$0] != "unknown" }
        if let g = given.max(by: { abs(m.values[$0]) < abs(m.values[$1]) }) {
            let v = m.values[g]
            if abs(v) >= 0.01 {
                let size = abs(v) < 0.05 ? "a small amount" : "a noticeable amount"
                parts.append(String(format: "%@ %@ the score by %@ (%+.2f).",
                                    MetadataExplanation.groups[g].capitalized, v > 0 ? "raised" : "lowered", size, v))
            } else {
                parts.append("The age, sex and body site you entered barely changed the score.")
            }
        }
        if r.lowConfidence { parts.append("The model is not confident about this image.") }
        return parts.joined(separator: " ")
    }
}
