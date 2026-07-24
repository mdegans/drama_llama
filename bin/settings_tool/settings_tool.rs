/// A simple tool to test the settings GUI. It can be used to generate a TOML
/// representation of the settings but is mostly just a testbed for the GUI.
use drama_llama::PredictOptions;
use serde::Serialize;

#[derive(Clone, Copy, PartialEq, Default)]
enum Mode {
    #[default]
    Json,
    #[cfg(feature = "toml")]
    Toml,
}

impl Mode {
    fn render<S>(self, s: &S) -> String
    where
        S: Serialize,
    {
        match self {
            Mode::Json => match serde_json::to_string_pretty(&s) {
                Ok(s) => s,
                Err(e) => format!("Error: {}", e),
            },
            #[cfg(feature = "toml")]
            Mode::Toml => match toml::to_string_pretty(&s) {
                Ok(s) => s,
                Err(e) => format!("Error: {}", e),
            },
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Mode::Json => "JSON",
            #[cfg(feature = "toml")]
            Mode::Toml => "TOML",
        }
    }
}

#[derive(Default)]
struct App {
    pub options: PredictOptions,
    pub mode: Mode,
}

/// Cap for the n-tokens drag widget. There's no model loaded here to
/// derive a real context size from, so use a generous testbed value.
const MAX_CONTEXT_SIZE: usize = 131072;

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::left("settings")
            .default_size(400.0)
            .show_inside(ui, |ui| self.options.draw(ui, MAX_CONTEXT_SIZE));

        egui::CentralPanel::default().show_inside(ui, |ui| {
            egui::ComboBox::from_label("Format")
                .selected_text(self.mode.as_str())
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.mode,
                        Mode::Json,
                        Mode::Json.as_str(),
                    );
                    #[cfg(feature = "toml")]
                    ui.selectable_value(
                        &mut self.mode,
                        Mode::Toml,
                        Mode::Toml.as_str(),
                    );
                });

            ui.separator();
            ui.label(self.mode.render(&self.options))
        });
    }
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    eframe::run_native(
        "`drama_llama` Settings Tool",
        eframe::NativeOptions::default(),
        Box::new(|_| Ok(Box::new(App::default()))),
    )?;

    Ok(())
}
