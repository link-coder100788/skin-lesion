import Charts
import SwiftUI

struct ResultView: View {
    let analysis: Analysis
    @Environment(AppModel.self) private var model
    @State private var map = MapKind.shap

    enum MapKind: String, CaseIterable { case shap = "Image SHAP", gradcam = "Grad-CAM" }

    var body: some View {
        Group {
            switch analysis.phase {
            case .running:
                ProgressView("Analyzing on device…")
            case .failed(let msg):
                ContentUnavailableView("Analysis Failed", systemImage: "exclamationmark.triangle", description: Text(msg))
            case .done:
                if let r = analysis.result, let cfg = model.config { content(r, cfg) }
            }
        }
        .navigationTitle("Result")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func content(_ r: TriageResult, _ cfg: PipelineConfig) -> some View {
        ScrollViewReader { proxy in
        List {
            Section { VerdictCard(result: r) }

            Section {
                Picker("Map", selection: $map) {
                    ForEach(MapKind.allCases, id: \.self) { Text($0.rawValue) }
                }
                .pickerStyle(.segmented)
                .listRowSeparator(.hidden)
                mapView
                    .listRowSeparator(.hidden)
                if let s = analysis.summary {
                    Text(s).font(.callout)
                }
            } header: {
                Text("Where the model looked")
            } footer: {
                Text(map == .shap
                     ? "Red regions pushed the score toward \u{201C}see a doctor\u{201D}, blue regions away from it. These maps change under small edits to the photo: read them as hints, not proof."
                     : "Grad-CAM highlights where the network's last layers responded most (unsigned).")
            }

            Section {
                ProbabilityChart(probs: r.probs, classes: cfg.classes, urgent: cfg.urgent)
                    .frame(height: 220)
                    .opacity(r.unlikeTraining ? 0.4 : 1)
            } header: {
                Text("Most likely types")
            } footer: {
                if r.unlikeTraining {
                    Text("Not meaningful for this image: the model always picks one of its seven skin types, even for a photo that isn't skin.")
                }
            }

            Section {
                WaterfallChart(explanation: r.metadata)
                    .frame(height: 200)
            } header: {
                Text("How age, sex and site moved the score")
            } footer: {
                Text(cfg.useFusion ? "" : "The decision uses the image alone: adding age, sex and site did not improve accuracy on held-out data. This chart shows how they would shift the score.")
            }

            Section {
                Text(TriageText.disclaimer).font(.footnote).foregroundStyle(.secondary)
            }
            .id("end")
        }
        #if DEBUG
        // `-scrollEnd` scrolls to the bottom once the explanation is ready (for screenshots).
        .onChange(of: analysis.summary) {
            if ProcessInfo.processInfo.arguments.contains("-scrollEnd") { proxy.scrollTo("end", anchor: .bottom) }
        }
        #endif
        }
    }

    @ViewBuilder private var mapView: some View {
        let image = map == .shap ? analysis.shapImage : analysis.camImage
        if let image {
            Image(decorative: image, scale: 1)
                .resizable()
                .interpolation(.high)
                .scaledToFit()
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .frame(maxWidth: .infinity)
        } else if map == .shap {
            VStack(spacing: 10) {
                ProgressView(value: analysis.shapProgress)
                Text("Computing image SHAP on device…").font(.footnote).foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, minHeight: 120)
        } else {
            ContentUnavailableView {
                Label("Examples Only", systemImage: "square.stack.3d.down.right")
            } description: {
                Text("Grad-CAM needs gradients, which Core ML can't compute on the phone. It was precomputed for the bundled examples.")
            }
        }
    }
}

private struct VerdictCard: View {
    let result: TriageResult

    private var style: (title: String, symbol: String, color: Color, advice: String) {
        if result.unlikeTraining {
            return ("Cannot assess this image", "questionmark.circle.fill", .gray, TriageText.unlikeAdvice)
        }
        if result.urgent {
            return ("See a doctor soon", "exclamationmark.circle.fill", .red, TriageText.urgentAdvice)
        }
        return ("Lower concern", "checkmark.circle.fill", .green, TriageText.benignAdvice)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label {
                Text(style.title).font(.title2.bold())
            } icon: {
                Image(systemName: style.symbol).foregroundStyle(style.color)
            }
            Text(style.advice).font(.subheadline)
            if !result.unlikeTraining {
                VStack(alignment: .leading, spacing: 4) {
                    ScoreBar(score: Double(result.pUrgent), threshold: Double(result.tau), color: style.color)
                        .frame(height: 10)
                    Text(String(format: "Urgent score %.2f · threshold %.2f (tick)", result.pUrgent, result.tau))
                        .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                }
            }
            if result.lowConfidence && !result.unlikeTraining {
                Label("The model is not confident about this image.", systemImage: "exclamationmark.triangle")
                    .font(.footnote)
                    .foregroundStyle(.orange)
            }
        }
        .padding(.vertical, 6)
        .accessibilityElement(children: .combine)
    }
}

private struct ScoreBar: View {
    let score: Double, threshold: Double, color: Color

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(.quaternary)
                Capsule().fill(color.gradient).frame(width: max(8, geo.size.width * score))
                Rectangle().fill(.primary).frame(width: 2, height: geo.size.height + 6)
                    .offset(x: geo.size.width * threshold - 1)
            }
        }
    }
}

private struct ProbabilityChart: View {
    let probs: [Float], classes: [String], urgent: [String]

    var body: some View {
        let rows = classes.indices.sorted { probs[$0] > probs[$1] }
        Chart(rows, id: \.self) { i in
            BarMark(x: .value("Probability", probs[i]), y: .value("Type", Lesion.short[classes[i]] ?? classes[i]))
                .foregroundStyle(by: .value("Group", urgent.contains(classes[i]) ? "Urgent type" : "Other"))
                .annotation(position: .trailing) {
                    Text(String(format: "%.0f%%", probs[i] * 100)).font(.caption2).foregroundStyle(.secondary)
                }
        }
        .chartForegroundStyleScale(["Urgent type": Color.red, "Other": Color.gray])
        .chartXScale(domain: 0...1.1)
        .chartXAxis(.hidden)
        .chartLegend(position: .bottom)
    }
}

/// Base rate -> image -> age -> sex -> body site, in probability units.
private struct WaterfallChart: View {
    let explanation: MetadataExplanation

    private struct Step: Identifiable {
        let id: Int, name: String, start: Double, end: Double
    }

    var body: some View {
        var steps: [Step] = []
        var x = explanation.base
        let names = ["Image", "Age", "Sex", "Body site"]
        for (i, name) in names.enumerated() {
            let v = explanation.values[i]
            let label = "\(name): \(explanation.shown[i])"
            steps.append(Step(id: i, name: label, start: x, end: x + v))
            x += v
        }
        return Chart {
            ForEach(steps) { s in
                BarMark(xStart: .value("From", s.start), xEnd: .value("To", s.end), y: .value("Input", s.name))
                    .foregroundStyle(s.end >= s.start ? Color.red : Color.blue)
                    .annotation(position: s.end >= s.start ? .trailing : .leading) {
                        Text(String(format: "%+.3f", s.end - s.start))
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
            }
            RuleMark(x: .value("Average", explanation.base))
                .lineStyle(StrokeStyle(lineWidth: 1, dash: [3]))
                .foregroundStyle(.secondary)
                .annotation(position: .top, alignment: .leading) {
                    Text(String(format: "average %.2f", explanation.base)).font(.caption2).foregroundStyle(.secondary)
                }
        }
        .chartXScale(domain: -0.1...1.15)
        .chartXAxisLabel("Urgent score")
    }
}
