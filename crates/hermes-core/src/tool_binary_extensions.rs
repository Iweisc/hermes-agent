//! Binary file extensions to skip for text-based operations.
//!
//! These files can't be meaningfully compared as text and are often large.
//! Ported from `tools/binary_extensions.py` (originally from free-code
//! `src/constants/files.ts`).

/// The set of file extensions (including the leading dot, lowercase) that are
/// considered binary and should be skipped for text-based operations.
///
/// Note: `.pdf` is intentionally excluded — it is text-adjacent and agents may
/// want to inspect it.
pub static BINARY_EXTENSIONS: &[&str] = &[
    // Images
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".tif",
    // Videos
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv", ".m4v", ".mpeg", ".mpg",
    // Audio
    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".aiff", ".opus",
    // Archives
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".xz", ".z", ".tgz", ".iso",
    // Executables/binaries
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".obj", ".lib",
    ".app", ".msi", ".deb", ".rpm",
    // Documents (exclude .pdf — text-based, agents may want to inspect)
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".odt", ".ods", ".odp",
    // Fonts
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    // Bytecode / VM artifacts
    ".pyc", ".pyo", ".class", ".jar", ".war", ".ear", ".node", ".wasm", ".rlib",
    // Database files
    ".sqlite", ".sqlite3", ".db", ".mdb", ".idx",
    // Design / 3D
    ".psd", ".ai", ".eps", ".sketch", ".fig", ".xd", ".blend", ".3ds", ".max",
    // Flash
    ".swf", ".fla",
    // Lock/profiling data
    ".lockb", ".dat", ".data",
];

/// Check whether `path` ends with the given lowercase extension.
///
/// This mirrors the Python behavior: the extension is taken to be everything
/// from the last `.` to the end of the string, lowercased, and compared
/// against the known set. Pure string check, no I/O.
fn extension_is_binary(ext_lower: &str) -> bool {
    BINARY_EXTENSIONS.iter().any(|&e| e == ext_lower)
}

/// Check if a file path has a binary extension. Pure string check, no I/O.
///
/// The "extension" is everything from the last `.` in the path to the end,
/// matching Python's `path.rfind(".")` semantics. If there is no `.`, returns
/// `false`. The comparison is case-insensitive.
pub fn has_binary_extension(path: &str) -> bool {
    // Mirror Python's str.rfind(".") which operates on the whole path,
    // not just the final path component.
    match path.rfind('.') {
        None => false,
        Some(dot) => {
            // path[dot:] -> the dot and everything after it.
            let ext = &path[dot..];
            extension_is_binary(&ext.to_lowercase())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_common_binary_extensions() {
        assert!(has_binary_extension("foo.png"));
        assert!(has_binary_extension("a/b/c.mp4"));
        assert!(has_binary_extension("archive.tar.gz"));
        assert!(has_binary_extension("lib.so"));
        assert!(has_binary_extension("data.sqlite3"));
    }

    #[test]
    fn case_insensitive() {
        assert!(has_binary_extension("IMAGE.PNG"));
        assert!(has_binary_extension("Movie.MoV"));
        assert!(has_binary_extension("FONT.WOFF2"));
    }

    #[test]
    fn rejects_text_extensions() {
        assert!(!has_binary_extension("main.rs"));
        assert!(!has_binary_extension("README.md"));
        assert!(!has_binary_extension("notes.txt"));
        // .pdf is intentionally NOT binary.
        assert!(!has_binary_extension("report.pdf"));
    }

    #[test]
    fn no_dot_returns_false() {
        assert!(!has_binary_extension("Makefile"));
        assert!(!has_binary_extension("LICENSE"));
        assert!(!has_binary_extension(""));
    }

    #[test]
    fn uses_last_dot_like_python() {
        // The extension is everything after the LAST dot.
        assert!(has_binary_extension("foo.txt.png"));
        assert!(!has_binary_extension("foo.png.txt"));
    }

    #[test]
    fn dotfile_with_binary_name() {
        // Python: path=".png", rfind(".")==0, path[0:]==".png" -> binary.
        assert!(has_binary_extension(".png"));
        // A dotfile that is not a known binary extension.
        assert!(!has_binary_extension(".gitignore"));
    }

    #[test]
    fn trailing_dot() {
        // path="foo." -> path[dot:]=="." which is not in the set.
        assert!(!has_binary_extension("foo."));
    }

    #[test]
    fn set_contains_expected_count() {
        // Guard against accidental edits; the Python frozenset has 93 entries.
        assert_eq!(BINARY_EXTENSIONS.len(), 93);
    }
}
