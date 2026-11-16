import SwiftUI

/// Model card: what the model is, how well it did on the locked test set, and where it fails.
struct AboutView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Label(TriageText.disclaimer, systemImage: "exclamationmark.triangle.fill")
                        .symbolRenderingMode(.multicolor)
                }
                if let cfg = model.config {
                    Section("Model") {
                        LabeledContent("Network", value: cfg.model)
                        LabeledContent("Urgent types", value: "Melanoma, basal cell carcinoma, actinic keratosis")
                        LabeledContent("Threshold", value: String(format: "%.3f", cfg.tau))
                        LabeledContent("Runs on", value: "This iPhone (Core ML)")
                    }
                    Section {
                        metric("Urgent sensitivity", "sensitivity_urgent", cfg)
                        metric("Specificity", "specificity_urgent", cfg)
                        metric("Urgent AUROC", "auroc_urgent", cfg)
                        metric("Balanced accuracy (7 types)", "balanced_accuracy", cfg)
                    } header: {
                        Text("Locked test set · 1,503 HAM10000 images")
                    } footer: {
                        Text("The threshold was set for 90% sensitivity on validation, but caught 80% of urgent cases on the test set.")
                    }
                }
                Section("Known limits") {
                    limit("Clinical photos", "Trained on dermoscopy. On ordinary clinical photos it catches fewer cancers (69–77%) and raises far more false alarms.")
                    limit("Darker skin", "False alarms rise with darker skin types (specificity 0.63 for types I–II, 0.36 for V–VI on Fitzpatrick17k). HAM10000 has very little dark skin.")
                    limit("Unusual images", "Non-skin or unfamiliar photos get \u{201C}cannot assess\u{201D}. The guard caught 19 of 22 non-skin test images, not all.")
                    limit("Explanations", "Heatmaps are hints, not proof: they shift under small edits to the photo.")
                }
                Section("Privacy") {
                    Text("Photos are analysed on this iPhone. Nothing is uploaded or stored.")
                }
            }
            .navigationTitle("About")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
    }

    private func metric(_ name: String, _ key: String, _ cfg: PipelineConfig) -> some View {
        LabeledContent(name, value: String(format: "%.2f", cfg.testMetrics[key] ?? .nan))
            .monospacedDigit()
    }

    private func limit(_ title: String, _ body: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.subheadline.bold())
            Text(body).font(.footnote).foregroundStyle(.secondary)
        }
    }
}
