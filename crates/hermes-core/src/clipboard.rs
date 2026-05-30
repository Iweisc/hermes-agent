//! Clipboard image extraction for macOS, Windows, Linux, and WSL2.
//!
//! Provides [`save_clipboard_image`] which checks the system clipboard for
//! image data, saves it to a destination path as PNG, and returns `true` on
//! success. No external Rust dependencies beyond the `image` crate (for
//! BMP/TIFF -> PNG conversion and dimensions) — clipboard access goes through
//! OS-level CLI tools that ship with the platform (or are commonly installed).
//!
//! Platform support:
//!   macOS   — osascript (always available), pngpaste (if installed)
//!   Windows — PowerShell via WinForms, Get-Clipboard, file-drop fallback
//!   WSL2    — powershell.exe via WinForms, Get-Clipboard, file-drop fallback
//!   Linux   — wl-paste (Wayland), xclip (X11)

use std::env;
use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

use base64::Engine as _;

use crate::is_wsl;

/// Extract an image from the system clipboard and save it as PNG.
///
/// Returns `true` if an image was found and saved, `false` otherwise.
pub fn save_clipboard_image(dest: &Path) -> bool {
    if let Some(parent) = dest.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if cfg!(target_os = "macos") {
        macos_save(dest)
    } else if cfg!(target_os = "windows") {
        windows_save(dest)
    } else {
        linux_save(dest)
    }
}

/// Quick check: does the clipboard currently contain an image?
///
/// Lighter than [`save_clipboard_image`] — doesn't extract or write anything.
pub fn has_clipboard_image() -> bool {
    if cfg!(target_os = "macos") {
        return macos_has_image();
    }
    if cfg!(target_os = "windows") {
        return windows_has_image();
    }
    // Match linux_save fallthrough order: WSL -> Wayland -> X11
    if is_wsl() && wsl_has_image() {
        return true;
    }
    if env::var_os("WAYLAND_DISPLAY").is_some() && wayland_has_image() {
        return true;
    }
    xclip_has_image()
}

fn file_has_bytes(path: &Path) -> bool {
    fs::metadata(path).map(|m| m.len() > 0).unwrap_or(false)
}

// Run a command with a timeout, capturing stdout/stderr. Returns the output, or
// `None` if the binary is missing, the spawn failed, or it timed out.
fn run_capture(program: &str, args: &[&str], timeout: Duration) -> Option<std::process::Output> {
    let child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(child) => child,
        Err(_) => return None, // not installed / spawn failed
    };
    match wait_timeout(&mut child, timeout) {
        Some(true) => child.wait_with_output().ok(),
        Some(false) | None => {
            let _ = child.kill();
            let _ = child.wait();
            None
        }
    }
}

// Poll for process exit up to `timeout`. Returns Some(true) on clean exit,
// Some(false) on timeout, None on a wait error.
fn wait_timeout(child: &mut std::process::Child, timeout: Duration) -> Option<bool> {
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return Some(true),
            Ok(None) => {
                if start.elapsed() >= timeout {
                    return Some(false);
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(_) => return None,
        }
    }
}

// ── macOS ────────────────────────────────────────────────────────────────

fn macos_save(dest: &Path) -> bool {
    // Try pngpaste first (fast, handles more formats), fall back to osascript.
    macos_pngpaste(dest) || macos_osascript(dest)
}

fn macos_has_image() -> bool {
    let Some(output) = run_capture(
        "osascript",
        &["-e", "clipboard info"],
        Duration::from_secs(3),
    ) else {
        return false;
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.contains("\u{ab}class PNGf\u{bb}") || stdout.contains("\u{ab}class TIFF\u{bb}")
}

fn macos_pngpaste(dest: &Path) -> bool {
    let Some(output) = run_capture(
        "pngpaste",
        &[&dest.to_string_lossy()],
        Duration::from_secs(3),
    ) else {
        return false;
    };
    output.status.success() && file_has_bytes(dest)
}

fn macos_osascript(dest: &Path) -> bool {
    if !macos_has_image() {
        return false;
    }
    let script = format!(
        "try\n\
           set imgData to the clipboard as \u{ab}class PNGf\u{bb}\n\
           set f to open for access POSIX file \"{}\" with write permission\n\
           write imgData to f\n\
           close access f\n\
         on error\n\
           return \"fail\"\n\
         end try\n",
        dest.display()
    );
    let Some(output) = run_capture("osascript", &["-e", &script], Duration::from_secs(5)) else {
        return false;
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    output.status.success() && !stdout.contains("fail") && file_has_bytes(dest)
}

// ── Shared PowerShell scripts (native Windows + WSL2) ─────────────────────

const PS_CHECK_IMAGE: &str = "Add-Type -AssemblyName System.Windows.Forms;\
[System.Windows.Forms.Clipboard]::ContainsImage()";

const PS_EXTRACT_IMAGE: &str = "Add-Type -AssemblyName System.Windows.Forms;\
Add-Type -AssemblyName System.Drawing;\
$img = [System.Windows.Forms.Clipboard]::GetImage();\
if ($null -eq $img) { exit 1 }\
$ms = New-Object System.IO.MemoryStream;\
$img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png);\
[System.Convert]::ToBase64String($ms.ToArray())";

const PS_CHECK_IMAGE_GET_CLIPBOARD: &str = "try { \
$img = Get-Clipboard -Format Image -ErrorAction Stop;\
if ($null -ne $img) { 'True' } else { 'False' }\
} catch { 'False' }";

const PS_EXTRACT_IMAGE_GET_CLIPBOARD: &str = "try { \
Add-Type -AssemblyName System.Drawing;\
Add-Type -AssemblyName PresentationCore;\
Add-Type -AssemblyName WindowsBase;\
$img = Get-Clipboard -Format Image -ErrorAction Stop;\
if ($null -eq $img) { exit 1 }\
$ms = New-Object System.IO.MemoryStream;\
if ($img -is [System.Drawing.Image]) {\
$img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)\
} elseif ($img -is [System.Windows.Media.Imaging.BitmapSource]) {\
$enc = New-Object System.Windows.Media.Imaging.PngBitmapEncoder;\
$enc.Frames.Add([System.Windows.Media.Imaging.BitmapFrame]::Create($img));\
$enc.Save($ms)\
} else { exit 2 }\
[System.Convert]::ToBase64String($ms.ToArray())\
} catch { exit 1 }";

const FILEDROP_IMAGE_EXTS: &str =
    "'.png','.jpg','.jpeg','.gif','.webp','.bmp','.tiff','.tif'";

fn ps_check_filedrop_image() -> String {
    format!(
        "try {{ \
$files = Get-Clipboard -Format FileDropList -ErrorAction Stop;\
$exts = @({exts});\
$hit = $files | Where-Object {{ $exts -contains ([System.IO.Path]::GetExtension($_).ToLowerInvariant()) }} | Select-Object -First 1;\
if ($null -ne $hit) {{ 'True' }} else {{ 'False' }}\
}} catch {{ 'False' }}",
        exts = FILEDROP_IMAGE_EXTS
    )
}

fn ps_extract_filedrop_image() -> String {
    format!(
        "try {{ \
$files = Get-Clipboard -Format FileDropList -ErrorAction Stop;\
$exts = @({exts});\
$hit = $files | Where-Object {{ $exts -contains ([System.IO.Path]::GetExtension($_).ToLowerInvariant()) }} | Select-Object -First 1;\
if ($null -eq $hit) {{ exit 1 }}\
[System.Convert]::ToBase64String([System.IO.File]::ReadAllBytes($hit))\
}} catch {{ exit 1 }}",
        exts = FILEDROP_IMAGE_EXTS
    )
}

fn powershell_has_image_scripts() -> Vec<String> {
    vec![
        PS_CHECK_IMAGE.to_string(),
        PS_CHECK_IMAGE_GET_CLIPBOARD.to_string(),
        ps_check_filedrop_image(),
    ]
}

fn powershell_extract_image_scripts() -> Vec<String> {
    vec![
        PS_EXTRACT_IMAGE.to_string(),
        PS_EXTRACT_IMAGE_GET_CLIPBOARD.to_string(),
        ps_extract_filedrop_image(),
    ]
}

fn run_powershell(exe: &str, script: &str, timeout: Duration) -> Option<std::process::Output> {
    run_capture(
        exe,
        &["-NoProfile", "-NonInteractive", "-Command", script],
        timeout,
    )
}

fn write_base64_image(dest: &Path, b64_data: &str) -> bool {
    let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64_data) else {
        return false;
    };
    if fs::write(dest, &bytes).is_err() {
        return false;
    }
    file_has_bytes(dest)
}

fn powershell_has_image(exe: &str, timeout: Duration) -> bool {
    for script in powershell_has_image_scripts() {
        match run_powershell(exe, &script, timeout) {
            Some(output) => {
                if output.status.success()
                    && String::from_utf8_lossy(&output.stdout).contains("True")
                {
                    return true;
                }
            }
            // Binary missing (spawn failed) — clipboard unavailable.
            None => return false,
        }
    }
    false
}

fn powershell_save_image(exe: &str, dest: &Path, timeout: Duration) -> bool {
    for script in powershell_extract_image_scripts() {
        let Some(output) = run_powershell(exe, &script, timeout) else {
            // Binary missing (spawn failed) — clipboard unavailable.
            return false;
        };
        if !output.status.success() {
            continue;
        }
        let b64_data = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if b64_data.is_empty() {
            continue;
        }
        if write_base64_image(dest, &b64_data) {
            return true;
        }
        let _ = fs::remove_file(dest);
    }
    false
}

// ── Native Windows ─────────────────────────────────────────────────────────

// Native Windows uses `powershell` (Windows PowerShell 5.1, always present) or
// `pwsh` (PowerShell 7+, optional).
fn find_powershell() -> Option<&'static str> {
    for name in ["powershell", "pwsh"] {
        if let Some(output) = run_capture(
            name,
            &["-NoProfile", "-NonInteractive", "-Command", "echo ok"],
            Duration::from_secs(5),
        ) {
            if output.status.success() && String::from_utf8_lossy(&output.stdout).contains("ok") {
                return Some(if name == "powershell" {
                    "powershell"
                } else {
                    "pwsh"
                });
            }
        }
    }
    None
}

fn windows_has_image() -> bool {
    let Some(ps) = find_powershell() else {
        return false;
    };
    powershell_has_image(ps, Duration::from_secs(5))
}

fn windows_save(dest: &Path) -> bool {
    let Some(ps) = find_powershell() else {
        return false;
    };
    powershell_save_image(ps, dest, Duration::from_secs(15))
}

// ── Linux ────────────────────────────────────────────────────────────────

fn linux_save(dest: &Path) -> bool {
    // Try clipboard backends in priority order: WSL -> Wayland -> X11.
    if is_wsl() && wsl_save(dest) {
        return true;
        // Fall through — WSLg might have wl-paste or xclip working.
    }
    if env::var_os("WAYLAND_DISPLAY").is_some() && wayland_save(dest) {
        return true;
    }
    xclip_save(dest)
}

// ── WSL2 (powershell.exe) ────────────────────────────────────────────────

fn wsl_has_image() -> bool {
    powershell_has_image("powershell.exe", Duration::from_secs(8))
}

fn wsl_save(dest: &Path) -> bool {
    powershell_save_image("powershell.exe", dest, Duration::from_secs(15))
}

// ── Wayland (wl-paste) ──────────────────────────────────────────────────

fn wayland_list_types() -> Option<Vec<String>> {
    let output = run_capture("wl-paste", &["--list-types"], Duration::from_secs(3))?;
    if !output.status.success() {
        return None;
    }
    Some(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::to_string)
            .collect(),
    )
}

fn wayland_has_image() -> bool {
    wayland_list_types()
        .map(|types| types.iter().any(|t| t.starts_with("image/")))
        .unwrap_or(false)
}

fn wayland_save(dest: &Path) -> bool {
    let Some(types) = wayland_list_types() else {
        return false;
    };

    // Prefer PNG, fall back to other image formats.
    let mut mime: Option<&str> = None;
    for preferred in [
        "image/png",
        "image/jpeg",
        "image/bmp",
        "image/gif",
        "image/webp",
    ] {
        if types.iter().any(|t| t == preferred) {
            mime = Some(preferred);
            break;
        }
    }
    let Some(mime) = mime else {
        return false;
    };

    // Extract the image data.
    let Some(output) = run_capture("wl-paste", &["--type", mime], Duration::from_secs(5)) else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    if fs::write(dest, &output.stdout).is_err() || !file_has_bytes(dest) {
        let _ = fs::remove_file(dest);
        return false;
    }

    // BMP needs conversion to PNG (common in WSLg where only BMP is bridged
    // from the Windows clipboard via RDP).
    if mime == "image/bmp" {
        return convert_to_png(dest);
    }
    true
}

fn convert_to_png(path: &Path) -> bool {
    // Decode via the `image` crate and re-encode as PNG in place.
    match image::open(path) {
        Ok(img) => {
            if img
                .save_with_format(path, image::ImageFormat::Png)
                .is_ok()
                && file_has_bytes(path)
            {
                return true;
            }
        }
        Err(_) => {}
    }
    // Can't convert — the original is still usable as-is for most APIs.
    file_has_bytes(path)
}

// ── X11 (xclip) ─────────────────────────────────────────────────────────

fn xclip_targets_have_png() -> bool {
    let Some(output) = run_capture(
        "xclip",
        &["-selection", "clipboard", "-t", "TARGETS", "-o"],
        Duration::from_secs(3),
    ) else {
        return false;
    };
    output.status.success() && String::from_utf8_lossy(&output.stdout).contains("image/png")
}

fn xclip_has_image() -> bool {
    xclip_targets_have_png()
}

fn xclip_save(dest: &Path) -> bool {
    if !xclip_targets_have_png() {
        return false;
    }
    let Some(output) = run_capture(
        "xclip",
        &["-selection", "clipboard", "-t", "image/png", "-o"],
        Duration::from_secs(5),
    ) else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    if fs::write(dest, &output.stdout).is_err() || !file_has_bytes(dest) {
        let _ = fs::remove_file(dest);
        return false;
    }
    true
}

// ── Image metadata ─────────────────────────────────────────────────────────

/// Read the pixel dimensions of an image file, mirroring the Python
/// `Image.open(path).size` used by the paste helper. Returns `None` if the
/// file cannot be decoded.
pub fn image_dimensions(path: &Path) -> Option<(u32, u32)> {
    image::image_dimensions(path).ok()
}

/// Estimate the token cost of an image at the given dimensions, matching the
/// Python helper:
/// `max(1, (w + 511) // 512) * max(1, (h + 511) // 512) * 85`.
pub fn image_token_estimate(width: u32, height: u32) -> u64 {
    let tiles_w = std::cmp::max(1, (width as u64 + 511) / 512);
    let tiles_h = std::cmp::max(1, (height as u64 + 511) / 512);
    tiles_w * tiles_h * 85
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mirrors the Python helper:
    // max(1, (w + 511) // 512) * max(1, (h + 511) // 512) * 85
    fn python_estimate(w: u64, h: u64) -> u64 {
        std::cmp::max(1, (w + 511) / 512) * std::cmp::max(1, (h + 511) / 512) * 85
    }

    #[test]
    fn token_estimate_matches_python_formula() {
        for (w, h) in [(1, 1), (512, 512), (513, 100), (1024, 768), (0, 0), (2000, 1500)] {
            assert_eq!(
                image_token_estimate(w, h),
                python_estimate(w as u64, h as u64),
                "mismatch for {w}x{h}"
            );
        }
    }

    #[test]
    fn empty_clipboard_check_does_not_panic() {
        // Smoke test: querying the clipboard in a headless environment must
        // return cleanly (false) rather than panic.
        let _ = has_clipboard_image();
    }
}
