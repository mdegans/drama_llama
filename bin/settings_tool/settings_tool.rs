/// A simple tool to test the settings GUI. It can be used to generate a TOML
/// representation of the settings but is mostly just a testbed for the GUI.
use drama_llama::PredictOptions;
use rocket::serde::Serialize;

#[derive(Clone, Copy, PartialEq, Default)]
enum Mode {
    #[default]
    JSON,
    #[cfg(feature = "toml")]
    TOML,
}

impl Mode {
    fn render<S>(self, s: &S) -> String
    where
        S: Serialize,
    {
        match self {
            Mode::JSON => match serde_json::to_string_pretty(&s) {
                Ok(s) => s,
                Err(e) => format!("Error: {}", e),
            },
            #[cfg(feature = "toml")]
            Mode::TOML => match toml::to_string_pretty(&s) {
                Ok(s) => s,
                Err(e) => format!("Error: {}", e),
            },
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Mode::JSON => "JSON",
            #[cfg(feature = "toml")]
            Mode::TOML => "TOML",
        }
    }
}

#[derive(Default)]
struct App {
    pub options: PredictOptions,
    pub mode: Mode,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("settings")
            .default_width(400.0)
            .show(ctx, |ui| self.options.draw(ui));

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ComboBox::from_label("Format")
                .selected_text(self.mode.as_str())
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.mode,
                        Mode::JSON,
                        Mode::JSON.as_str(),
                    );
                    #[cfg(feature = "toml")]
                    ui.selectable_value(
                        &mut self.mode,
                        Mode::TOML,
                        Mode::TOML.as_str(),
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
        Box::new(|_| Box::new(App::default())),
    )?;

    Ok(())
}
