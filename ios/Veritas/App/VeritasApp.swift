import SwiftUI

@main
struct VeritasApp: App {
    @State private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            CheckView()
                .environment(model)
        }
    }
}
