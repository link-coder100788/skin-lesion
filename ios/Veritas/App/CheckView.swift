import PhotosUI
import SwiftUI

/// Home screen: pick a photo (or an example), optionally add age, sex and body site, analyze.
struct CheckView: View {
    @Environment(AppModel.self) private var model
    @State private var selection: Selection?
    @State private var photoItem: PhotosPickerItem?
    @State private var showCamera = false
    @State private var age: Int?
    @State private var sex = "unknown"
    @State private var site = "unknown"
    @State private var analysis: Analysis?
    @State private var showAbout = false

    private var info: PatientInfo {
        PatientInfo(age: age, sex: sex == "unknown" ? nil : sex, site: site == "unknown" ? nil : site)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Label(TriageText.disclaimer, systemImage: "exclamationmark.triangle.fill")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .symbolRenderingMode(.multicolor)
                }

                Section("Photo") {
                    photoPreview
                    HStack {
                        PhotosPicker(selection: $photoItem, matching: .images) {
                            Label("Photos", systemImage: "photo.on.rectangle")
                                .frame(maxWidth: .infinity)
                        }
                        if UIImagePickerController.isSourceTypeAvailable(.camera) {
                            Button {
                                showCamera = true
                            } label: {
                                Label("Camera", systemImage: "camera")
                                    .frame(maxWidth: .infinity)
                            }
                        }
                    }
                    .buttonStyle(.bordered)
                }

                Section {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 12) {
                            ForEach(model.examples) { ex in
                                ExampleTile(example: ex, isSelected: selection == .example(ex))
                                    .onTapGesture { pick(ex) }
                            }
                        }
                        .padding(.vertical, 4)
                    }
                    .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                } header: {
                    Text("Or try an example")
                } footer: {
                    Text("Dermoscopy images from the held-out HAM10000 test set (CC BY-NC-SA 4.0), plus one non-skin photo.")
                }

                Section {
                    Picker("Age", selection: $age) {
                        Text("Unknown").tag(Int?.none)
                        ForEach(Array(stride(from: 5, through: 90, by: 5)), id: \.self) { Text("\($0)").tag(Int?.some($0)) }
                    }
                    Picker("Sex", selection: $sex) {
                        Text("Unknown").tag("unknown")
                        Text("Female").tag("female")
                        Text("Male").tag("male")
                    }
                    Picker("Body site", selection: $site) {
                        Text("Unknown").tag("unknown")
                        ForEach(model.config?.sites ?? [], id: \.self) { Text($0.capitalized).tag($0) }
                    }
                } header: {
                    Text("About the person (optional)")
                } footer: {
                    Text("Used to explain how age, sex and body site shift the score. The decision itself uses the image.")
                }
            }
            .navigationTitle("Check a Spot")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("About", systemImage: "info.circle") { showAbout = true }
                }
            }
            .safeAreaInset(edge: .bottom) { analyzeButton }
            .navigationDestination(item: $analysis) { ResultView(analysis: $0) }
            .sheet(isPresented: $showAbout) { AboutView() }
            .fullScreenCover(isPresented: $showCamera) {
                CameraPicker { selection = .photo($0) }.ignoresSafeArea()
            }
            #if DEBUG
            // `-demo <example id>` opens that example's result on launch (for screenshots).
            .task(id: model.pipeline == nil) {
                let args = ProcessInfo.processInfo.arguments
                if let i = args.firstIndex(of: "-demo"), i + 1 < args.count,
                   model.pipeline != nil, analysis == nil,
                   let ex = model.examples.first(where: { $0.id == args[i + 1] }) {
                    pick(ex)
                    analysis = model.analyze(.example(ex), info: info)
                }
            }
            #endif
            .onChange(of: photoItem) { _, item in
                Task {
                    if let data = try? await item?.loadTransferable(type: Data.self), let img = UIImage(data: data) {
                        selection = .photo(img)
                    }
                }
            }
        }
    }

    @ViewBuilder private var photoPreview: some View {
        if let selection {
            Image(uiImage: selection.image)
                .resizable()
                .scaledToFit()
                .frame(maxWidth: .infinity, maxHeight: 260)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .listRowInsets(EdgeInsets(top: 12, leading: 16, bottom: 12, trailing: 16))
        } else {
            ContentUnavailableView {
                Label("No Photo", systemImage: "camera.macro")
            } description: {
                Text("Use a close-up dermoscopy photo of a single skin spot.")
            }
        }
    }

    private var analyzeButton: some View {
        Button {
            if let selection { analysis = model.analyze(selection, info: info) }
        } label: {
            Group {
                if model.pipeline == nil && model.loadError == nil {
                    HStack { ProgressView(); Text("Loading model…") }
                } else {
                    Text("Analyze")
                }
            }
            .font(.headline)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 6)
        }
        .buttonStyle(.borderedProminent)
        .disabled(selection == nil || model.pipeline == nil)
        .padding(.horizontal)
        .padding(.bottom, 8)
        .background(.bar)
    }

    private func pick(_ ex: Example) {
        selection = .example(ex)
        photoItem = nil
        age = ex.age
        sex = ex.sex
        site = (model.config?.sites ?? []).contains(ex.site) ? ex.site : "unknown"
    }
}

extension Analysis: Hashable {
    static func == (a: Analysis, b: Analysis) -> Bool { a === b }
    func hash(into h: inout Hasher) { h.combine(ObjectIdentifier(self)) }
}

private struct ExampleTile: View {
    let example: Example
    let isSelected: Bool

    var body: some View {
        VStack(spacing: 6) {
            Image(uiImage: example.image)
                .resizable()
                .scaledToFill()
                .frame(width: 84, height: 84)
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .overlay {
                    RoundedRectangle(cornerRadius: 10)
                        .strokeBorder(isSelected ? Color.accentColor : .clear, lineWidth: 3)
                }
            Text(example.why == "dark-corner artifact" ? "Dark corners" : example.title)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .multilineTextAlignment(.center)
                .frame(width: 84, height: 28, alignment: .top)
        }
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }
}

/// Minimal camera wrapper (SwiftUI has no camera view of its own).
struct CameraPicker: UIViewControllerRepresentable {
    let onPick: (UIImage) -> Void
    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ vc: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: CameraPicker
        init(_ parent: CameraPicker) { self.parent = parent }

        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            if let img = info[.originalImage] as? UIImage { parent.onPick(img) }
            parent.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) { parent.dismiss() }
    }
}
