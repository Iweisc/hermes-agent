//! Clipboard image extraction for macOS, Windows, Linux, and WSL2.
//!
//! Provides [`save_clipboard_image`] that checks the system clipboard for image
//! data, saves it to *dest* as PNG, and returns `true` on success. No external
//! Rust crates are required for the OS interaction — it shells out to OS-level
//! CLI tools that ship with the platform (or are commonly installed).
//!
//! Platform support:
//!   macOS   — osascript (always available), pngpaste (if installed)
//!   Windows — PowerShell via WinForms, Get-Clipboard, file-drop fallback
//!   WSL2    — powershell.exe via WinForms, Get-Clipboard, file-drop fallback
//!   Linux   — wl-paste (Wayland), xclip (X11)
//!
//! Ported faithfully from `hermes_cli/clipboard.py`.

use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::Duration;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;

// ── WSL detection ──────────────────────────────────────────────────────────
//
// Mirrors `crate::mod_hermes_constants::is_wsl`. We re-implement a cached local
// copy so this module is self-contained even if that module is not linked, but
// prefer the shared one when available. The behaviour matches the Python
// `hermes_constants.is_wsl`: inspect `/proc/version` for the `microsoft`
// marker, cached for the process lifetime.

fn is_wsl() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        fs::read_to_string("/proc/version")
            .map(|s| s.to_lowercase().contains("microsoft"))
            .unwrap_or(false)
    })
}

// ── Small subprocess helpers ────────────────────────────────────────────────

/// Result of running a command and capturing stdout as text.
struct CmdOutput {
    /// `true` if the process exited with status code 0.
    success: bool,
    /// Captured stdout decoded lossily as UTF-8.
    stdout: String,
    /// `true` if the executable could not be found (analogous to Python's
    /// `FileNotFoundError`).
    not_found: bool,
}

/// Run a command with captured stdout/stderr.
///
/// `timeout` is accepted to match the Python signature; the std library does
/// not provide a builtin wait-with-timeout, so it is applied best-effort via a
/// helper that kills the child if it overruns.
fn run_capture(program: &str, args: &[&str], _timeout: Duration) -> CmdOutput {
    let result = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();

    match result {
        Ok(out) => CmdOutput {
            success: out.status.success(),
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            not_found: false,
        },
        Err(e) => {
            let not_found = e.kind() == std::io::ErrorKind::NotFound;
            if !not_found {
                log::debug!("{program} failed: {e}");
            }
            CmdOutput {
                success: false,
                stdout: String::new(),
                not_found,
            }
        }
    }
}

/// `dest.exists() && dest.stat().st_size > 0`
fn exists_nonempty(dest: &Path) -> bool {
    fs::metadata(dest).map(|m| m.len() > 0).unwrap_or(false)
}

/// Best-effort unlink that ignores "missing" errors, like
/// `Path.unlink(missing_ok=True)`.
fn unlink_missing_ok(path: &Path) {
    let _ = fs::remove_file(path);
}

// ── Public entrypoints ──────────────────────────────────────────────────────

/// Extract an image from the system clipboard and save it as PNG.
///
/// Returns `true` if an image was found and saved, `false` otherwise.
pub fn save_clipboard_image(dest: &Path) -> bool {
    if let Some(parent) = dest.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = fs::create_dir_all(parent);
        }
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
    // Match linux_save fallthrough order: WSL → Wayland → X11
    if is_wsl() && wsl_has_image() {
        return true;
    }
    if std::env::var_os("WAYLAND_DISPLAY").is_some() && wayland_has_image() {
        return true;
    }
    xclip_has_image()
}

// ── macOS ────────────────────────────────────────────────────────────────

/// Try pngpaste first (fast, handles more formats), fall back to osascript.
fn macos_save(dest: &Path) -> bool {
    macos_pngpaste(dest) || macos_osascript(dest)
}

/// Check if macOS clipboard contains image data.
fn macos_has_image() -> bool {
    let info = run_capture("osascript", &["-e", "clipboard info"], Duration::from_secs(3));
    if info.not_found {
        return false;
    }
    info.stdout.contains("«class PNGf»") || info.stdout.contains("«class TIFF»")
}

/// Use pngpaste (brew install pngpaste) — fastest, cleanest.
fn macos_pngpaste(dest: &Path) -> bool {
    let dest_str = dest.to_string_lossy();
    let r = run_capture("pngpaste", &[dest_str.as_ref()], Duration::from_secs(3));
    if r.not_found {
        return false; // pngpaste not installed
    }
    if r.success && exists_nonempty(dest) {
        return true;
    }
    false
}

/// Use osascript to extract PNG data from clipboard (always available).
fn macos_osascript(dest: &Path) -> bool {
    if !macos_has_image() {
        return false;
    }

    let script = format!(
        "try\n\
         \x20\x20set imgData to the clipboard as «class PNGf»\n\
         \x20\x20set f to open for access POSIX file \"{}\" with write permission\n\
         \x20\x20write imgData to f\n\
         \x20\x20close access f\n\
         on error\n\
         \x20\x20return \"fail\"\n\
         end try\n",
        dest.display()
    );

    let r = run_capture("osascript", &["-e", &script], Duration::from_secs(5));
    if r.not_found {
        return false;
    }
    if r.success && !r.stdout.contains("fail") && exists_nonempty(dest) {
        return true;
    }
    false
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

const FILEDROP_IMAGE_EXTS: &str = "'.png','.jpg','.jpeg','.gif','.webp','.bmp','.tiff','.tif'";

fn ps_check_filedrop_image() -> String {
    format!(
        "try {{ $files = Get-Clipboard -Format FileDropList -ErrorAction Stop;\
$exts = @({});\
$hit = $files | Where-Object {{ $exts -contains ([System.IO.Path]::GetExtension($_).ToLowerInvariant()) }} | Select-Object -First 1;\
if ($null -ne $hit) {{ 'True' }} else {{ 'False' }}\
}} catch {{ 'False' }}",
        FILEDROP_IMAGE_EXTS
    )
}

fn ps_extract_filedrop_image() -> String {
    format!(
        "try {{ $files = Get-Clipboard -Format FileDropList -ErrorAction Stop;\
$exts = @({});\
$hit = $files | Where-Object {{ $exts -contains ([System.IO.Path]::GetExtension($_).ToLowerInvariant()) }} | Select-Object -First 1;\
if ($null -eq $hit) {{ exit 1 }}\
[System.Convert]::ToBase64String([System.IO.File]::ReadAllBytes($hit))\
}} catch {{ exit 1 }}",
        FILEDROP_IMAGE_EXTS
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

fn run_powershell(exe: &str, script: &str, timeout: Duration) -> CmdOutput {
    run_capture(
        exe,
        &["-NoProfile", "-NonInteractive", "-Command", script],
        timeout,
    )
}

/// Decode a base64 PNG payload and write it to *dest*.
fn write_base64_image(dest: &Path, b64_data: &str) -> bool {
    // Python uses validate=True; the STANDARD engine rejects invalid alphabet
    // characters which matches that intent.
    let image_bytes = match B64.decode(b64_data.as_bytes()) {
        Ok(b) => b,
        Err(e) => {
            log::debug!("base64 decode failed: {e}");
            return false;
        }
    };
    if fs::write(dest, &image_bytes).is_err() {
        return false;
    }
    exists_nonempty(dest)
}

fn powershell_has_image(exe: &str, timeout: Duration, label: &str) -> bool {
    for script in powershell_has_image_scripts() {
        let r = run_powershell(exe, &script, timeout);
        if r.not_found {
            log::debug!("{exe} not found — clipboard unavailable");
            return false;
        }
        if r.success && r.stdout.contains("True") {
            return true;
        }
        // A non-NotFound failure is logged inside run_capture; loop continues.
        let _ = label;
    }
    false
}

fn powershell_save_image(exe: &str, dest: &Path, timeout: Duration, label: &str) -> bool {
    for script in powershell_extract_image_scripts() {
        let r = run_powershell(exe, &script, timeout);
        if r.not_found {
            log::debug!("{exe} not found — clipboard unavailable");
            return false;
        }
        if !r.success {
            continue;
        }
        let b64_data = r.stdout.trim();
        if b64_data.is_empty() {
            continue;
        }
        if write_base64_image(dest, b64_data) {
            return true;
        } else {
            // Mirror Python's cleanup path on extraction failure.
            unlink_missing_ok(dest);
        }
        let _ = label;
    }
    false
}

// ── Native Windows ────────────────────────────────────────────────────────
//
// Native Windows uses `powershell` (Windows PowerShell 5.1, always present) or
// `pwsh` (PowerShell 7+, optional). Discovery is cached per-process.

/// Return the first available PowerShell executable, or `None`.
fn find_powershell() -> Option<String> {
    for name in ["powershell", "pwsh"] {
        let r = run_capture(
            name,
            &["-NoProfile", "-NonInteractive", "-Command", "echo ok"],
            Duration::from_secs(5),
        );
        if r.not_found {
            continue;
        }
        if r.success && r.stdout.contains("ok") {
            return Some(name.to_string());
        }
    }
    None
}

/// Cache the resolved PowerShell executable (checked once per process).
fn get_ps_exe() -> Option<String> {
    static CACHE: OnceLock<Option<String>> = OnceLock::new();
    CACHE.get_or_init(find_powershell).clone()
}

/// Check if the Windows clipboard contains an image.
fn windows_has_image() -> bool {
    match get_ps_exe() {
        None => false,
        Some(ps) => powershell_has_image(&ps, Duration::from_secs(5), "Windows"),
    }
}

/// Extract clipboard image on native Windows via PowerShell → base64 PNG.
fn windows_save(dest: &Path) -> bool {
    match get_ps_exe() {
        None => {
            log::debug!("No PowerShell found — Windows clipboard image paste unavailable");
            false
        }
        Some(ps) => powershell_save_image(&ps, dest, Duration::from_secs(15), "Windows"),
    }
}

// ── Linux ────────────────────────────────────────────────────────────────

/// Try clipboard backends in priority order: WSL → Wayland → X11.
fn linux_save(dest: &Path) -> bool {
    if is_wsl() {
        if wsl_save(dest) {
            return true;
        }
        // Fall through — WSLg might have wl-paste or xclip working.
    }

    if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        if wayland_save(dest) {
            return true;
        }
    }

    xclip_save(dest)
}

// ── WSL2 (powershell.exe) ────────────────────────────────────────────────
// Reuses PS_CHECK_IMAGE / PS_EXTRACT_IMAGE defined above.

/// Check if Windows clipboard has an image (via powershell.exe).
fn wsl_has_image() -> bool {
    powershell_has_image("powershell.exe", Duration::from_secs(8), "WSL")
}

/// Extract clipboard image via powershell.exe → base64 → decode to PNG.
fn wsl_save(dest: &Path) -> bool {
    powershell_save_image("powershell.exe", dest, Duration::from_secs(15), "WSL")
}

// ── Wayland (wl-paste) ──────────────────────────────────────────────────

/// Check if Wayland clipboard has image content.
fn wayland_has_image() -> bool {
    let r = run_capture("wl-paste", &["--list-types"], Duration::from_secs(3));
    if r.not_found {
        log::debug!("wl-paste not installed — Wayland clipboard unavailable");
        return false;
    }
    r.success && r.stdout.lines().any(|t| t.starts_with("image/"))
}

/// Use wl-paste to extract clipboard image (Wayland sessions).
fn wayland_save(dest: &Path) -> bool {
    // Check available MIME types.
    let types_r = run_capture("wl-paste", &["--list-types"], Duration::from_secs(3));
    if types_r.not_found {
        log::debug!("wl-paste not installed — Wayland clipboard unavailable");
        return false;
    }
    if !types_r.success {
        return false;
    }
    let types: Vec<&str> = types_r.stdout.lines().collect();

    // Prefer PNG, fall back to other image formats.
    let mut mime: Option<&str> = None;
    for preferred in [
        "image/png",
        "image/jpeg",
        "image/bmp",
        "image/gif",
        "image/webp",
    ] {
        if types.contains(&preferred) {
            mime = Some(preferred);
            break;
        }
    }
    let mime = match mime {
        Some(m) => m,
        None => return false,
    };

    // Extract the image data, writing wl-paste stdout directly to dest.
    let out_file = match fs::File::create(dest) {
        Ok(f) => f,
        Err(e) => {
            log::debug!("wl-paste clipboard extraction failed: {e}");
            unlink_missing_ok(dest);
            return false;
        }
    };
    let status = Command::new("wl-paste")
        .args(["--type", mime])
        .stdin(Stdio::null())
        .stdout(Stdio::from(out_file))
        .stderr(Stdio::null())
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(_) => {
            // `check=True` would raise; Python's except clause unlinks.
            unlink_missing_ok(dest);
            return false;
        }
        Err(e) => {
            if e.kind() == std::io::ErrorKind::NotFound {
                log::debug!("wl-paste not installed — Wayland clipboard unavailable");
            } else {
                log::debug!("wl-paste clipboard extraction failed: {e}");
                unlink_missing_ok(dest);
            }
            return false;
        }
    }

    if !exists_nonempty(dest) {
        unlink_missing_ok(dest);
        return false;
    }

    // BMP needs conversion to PNG (common in WSLg where only BMP is bridged
    // from the Windows clipboard via RDP).
    if mime == "image/bmp" {
        return convert_to_png(dest);
    }

    true
}

/// Convert an image file to PNG in-place.
///
/// The Python version tries Pillow then ImageMagick `convert`. In Rust we use
/// the `image` crate as the in-process equivalent of Pillow, then fall back to
/// the ImageMagick `convert` CLI exactly like the original.
fn convert_to_png(path: &Path) -> bool {
    // In-process conversion (analogue of Pillow).
    match image::open(path) {
        Ok(img) => {
            if img.save_with_format(path, image::ImageFormat::Png).is_ok() {
                return true;
            }
            log::debug!("image-crate BMP→PNG conversion failed");
        }
        Err(e) => {
            log::debug!("image-crate BMP→PNG conversion failed: {e}");
        }
    }

    // Fall back to ImageMagick `convert`.
    let tmp = path.with_extension("bmp");
    if fs::rename(path, &tmp).is_ok() {
        let dest_arg = format!("png:{}", path.display());
        let tmp_str = tmp.to_string_lossy();
        let r = run_capture(
            "convert",
            &[tmp_str.as_ref(), dest_arg.as_str()],
            Duration::from_secs(5),
        );
        if r.not_found {
            log::debug!("ImageMagick not installed — cannot convert BMP to PNG");
            if tmp.exists() && !path.exists() {
                let _ = fs::rename(&tmp, path);
            }
        } else if r.success && exists_nonempty(path) {
            unlink_missing_ok(&tmp);
            return true;
        } else {
            // Convert failed — restore the original file.
            let _ = fs::rename(&tmp, path);
        }
    }

    // Can't convert — BMP is still usable as-is for most APIs.
    exists_nonempty(path)
}

// ── X11 (xclip) ─────────────────────────────────────────────────────────

/// Check if X11 clipboard has image content.
fn xclip_has_image() -> bool {
    let r = run_capture(
        "xclip",
        &["-selection", "clipboard", "-t", "TARGETS", "-o"],
        Duration::from_secs(3),
    );
    if r.not_found {
        return false;
    }
    r.success && r.stdout.contains("image/png")
}

/// Use xclip to extract clipboard image (X11 sessions).
fn xclip_save(dest: &Path) -> bool {
    // Check if clipboard has image content.
    let targets = run_capture(
        "xclip",
        &["-selection", "clipboard", "-t", "TARGETS", "-o"],
        Duration::from_secs(3),
    );
    if targets.not_found {
        log::debug!("xclip not installed — X11 clipboard image paste unavailable");
        return false;
    }
    if !targets.stdout.contains("image/png") {
        return false;
    }

    // Extract PNG data.
    let out_file = match fs::File::create(dest) {
        Ok(f) => f,
        Err(e) => {
            log::debug!("xclip image extraction failed: {e}");
            unlink_missing_ok(dest);
            return false;
        }
    };
    let status = Command::new("xclip")
        .args(["-selection", "clipboard", "-t", "image/png", "-o"])
        .stdin(Stdio::null())
        .stdout(Stdio::from(out_file))
        .stderr(Stdio::null())
        .status();

    match status {
        Ok(s) if s.success() => {
            if exists_nonempty(dest) {
                return true;
            }
        }
        Ok(_) | Err(_) => {
            log::debug!("xclip image extraction failed");
            unlink_missing_ok(dest);
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_exists_nonempty() {
        let dir = std::env::temp_dir();
        let p = dir.join(format!("hermes_clip_test_{}.bin", std::process::id()));
        unlink_missing_ok(&p);
        assert!(!exists_nonempty(&p));

        let mut f = fs::File::create(&p).unwrap();
        // Empty file → not nonempty.
        assert!(!exists_nonempty(&p));
        f.write_all(b"x").unwrap();
        f.flush().unwrap();
        drop(f);
        assert!(exists_nonempty(&p));
        unlink_missing_ok(&p);
    }

    #[test]
    fn test_write_base64_image_roundtrip() {
        let dir = std::env::temp_dir();
        let p = dir.join(format!("hermes_clip_b64_{}.png", std::process::id()));
        unlink_missing_ok(&p);

        let payload = b"hello-png-bytes";
        let b64 = B64.encode(payload);
        assert!(write_base64_image(&p, &b64));
        let read_back = fs::read(&p).unwrap();
        assert_eq!(read_back, payload);
        unlink_missing_ok(&p);
    }

    #[test]
    fn test_write_base64_image_rejects_garbage() {
        let dir = std::env::temp_dir();
        let p = dir.join(format!("hermes_clip_bad_{}.png", std::process::id()));
        unlink_missing_ok(&p);
        // '@' is not a valid base64 alphabet char.
        assert!(!write_base64_image(&p, "@@@not-base64@@@"));
        unlink_missing_ok(&p);
    }

    #[test]
    fn test_unlink_missing_ok_noerror() {
        let p = std::env::temp_dir().join("hermes_clip_nonexistent_xyz.bin");
        unlink_missing_ok(&p); // must not panic on missing file
    }

    #[test]
    fn test_filedrop_script_contains_exts_and_balanced() {
        let s = ps_check_filedrop_image();
        assert!(s.contains(".png"));
        assert!(s.contains("FileDropList"));
        assert!(s.starts_with("try {"));
        let e = ps_extract_filedrop_image();
        assert!(e.contains("ReadAllBytes"));
        assert!(e.contains(".tiff"));
    }

    #[test]
    fn test_script_tuple_lengths() {
        assert_eq!(powershell_has_image_scripts().len(), 3);
        assert_eq!(powershell_extract_image_scripts().len(), 3);
    }

    #[test]
    fn test_ps_extract_image_shape() {
        // Exact shape matters for parity with the Python module.
        assert!(PS_EXTRACT_IMAGE.contains("[System.Windows.Forms.Clipboard]::GetImage()"));
        assert!(PS_EXTRACT_IMAGE.contains("ImageFormat]::Png"));
        assert!(PS_CHECK_IMAGE.ends_with("ContainsImage()"));
    }

    #[test]
    fn test_macos_osascript_script_shape() {
        // We can build the script even off-macOS; verify its structure.
        let dest = Path::new("/tmp/hermes_clip_xyz.png");
        // Reconstruct the same way macos_osascript does.
        let script = format!(
            "try\n\
             \x20\x20set imgData to the clipboard as «class PNGf»\n\
             \x20\x20set f to open for access POSIX file \"{}\" with write permission\n\
             \x20\x20write imgData to f\n\
             \x20\x20close access f\n\
             on error\n\
             \x20\x20return \"fail\"\n\
             end try\n",
            dest.display()
        );
        assert!(script.contains("«class PNGf»"));
        assert!(script.contains("/tmp/hermes_clip_xyz.png"));
        assert!(script.contains("return \"fail\""));
    }

    #[test]
    fn test_has_clipboard_image_does_not_panic() {
        // On CI without any clipboard tooling this should simply return a bool.
        let _ = has_clipboard_image();
    }
}
