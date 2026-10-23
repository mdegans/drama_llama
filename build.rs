//! Embeds the commit this crate was built from, for blallama's `/version`
//! route: `DRAMA_LLAMA_GIT_COMMIT` (a full SHA, or `unknown`) and
//! `DRAMA_LLAMA_GIT_DIRTY` (`true`, `false` or `unknown`).
//!
//! From a git checkout it asks git. From a crates.io package, where there is
//! no `.git`, it reads the `.cargo_vcs_info.json` that `cargo package` writes.
//! Neither needs a build dependency.

use std::path::Path;
use std::process::Command;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let manifest_dir = Path::new(&manifest_dir);

    let (commit, dirty) = match git(manifest_dir, &["rev-parse", "HEAD"]) {
        Some(commit) => {
            let dirty = git(
                manifest_dir,
                &["status", "--porcelain", "--untracked-files=no"],
            )
            .map(|status| if status.is_empty() { "false" } else { "true" });
            (commit, dirty.unwrap_or("unknown"))
        }
        None => vcs_info(manifest_dir)
            .map(|(commit, dirty)| {
                (commit, if dirty { "true" } else { "false" })
            })
            .unwrap_or_else(|| ("unknown".to_string(), "unknown")),
    };
    println!("cargo:rustc-env=DRAMA_LLAMA_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=DRAMA_LLAMA_GIT_DIRTY={dirty}");

    // Naming any file replaces cargo's default (rerun when anything in the
    // package changes), so name the sources as well as git's pointers: a
    // new commit moves HEAD's ref, and an edit makes the tree dirty.
    for path in ["build.rs", "Cargo.toml", "src", "bin"] {
        println!("cargo:rerun-if-changed={path}");
    }
    let mut git_paths = vec![
        "HEAD".to_string(),
        "packed-refs".to_string(),
        "index".to_string(),
    ];
    if let Some(head_ref) = git(manifest_dir, &["symbolic-ref", "-q", "HEAD"]) {
        git_paths.push(head_ref);
    }
    for name in git_paths {
        if let Some(path) =
            git(manifest_dir, &["rev-parse", "--git-path", &name])
        {
            let path = manifest_dir.join(path);
            if path.exists() {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
}

/// `git <args>` in `dir`, trimmed, or `None` when git is missing or fails
fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// The commit and dirtiness `cargo package` recorded, read without a JSON
/// parser: `"sha1": "<hex>"` and an optional `"dirty": true`
fn vcs_info(dir: &Path) -> Option<(String, bool)> {
    let text =
        std::fs::read_to_string(dir.join(".cargo_vcs_info.json")).ok()?;
    let after = text.split("\"sha1\"").nth(1)?;
    let commit: String = after
        .split('"')
        .nth(1)?
        .chars()
        .filter(char::is_ascii_hexdigit)
        .collect();
    let dirty = text.split("\"dirty\"").nth(1).is_some_and(|rest| {
        rest.trim_start_matches([':', ' ']).starts_with("true")
    });
    (!commit.is_empty()).then_some((commit, dirty))
}
