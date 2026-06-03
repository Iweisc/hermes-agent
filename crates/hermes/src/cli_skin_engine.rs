//! Hermes CLI skin/theme engine (native Rust port of `hermes_cli/skin_engine.py`).
//!
//! A data-driven skin system that lets users customize the CLI's visual
//! appearance. Skins are defined as YAML files in `~/.hermes/skins/` or as
//! built-in presets. No code changes are needed to add a new skin.
//!
//! This is a faithful port of the Python module: it reproduces the built-in
//! skin table, the merge-over-`default` resolution, user YAML skins, the active
//! skin cache, and the convenience helpers used by CLI modules (including the
//! `prompt_toolkit` style overrides).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;

use serde_json::Value as JsonValue;
use serde_yaml::Value as YamlValue;

/// Default tool prefix used when a skin omits one.
pub const DEFAULT_TOOL_PREFIX: &str = "┊";

// =============================================================================
// Skin data structure
// =============================================================================

/// Complete (resolved) skin configuration. Port of the Python `SkinConfig`
/// dataclass.
#[derive(Debug, Clone)]
pub struct SkinConfig {
    pub name: String,
    pub description: String,
    /// Hex colors for Rich markup keyed by name (e.g. `banner_title`).
    pub colors: BTreeMap<String, String>,
    /// Spinner customization. Stored as a JSON object so list/string shapes
    /// round-trip faithfully (mirrors the Python `Dict[str, Any]`).
    pub spinner: JsonValue,
    pub branding: BTreeMap<String, String>,
    pub tool_prefix: String,
    /// Per-tool emoji overrides.
    pub tool_emojis: BTreeMap<String, String>,
    /// Rich-markup ASCII art logo (replaces `HERMES_AGENT_LOGO`).
    pub banner_logo: String,
    /// Rich-markup hero art (replaces `HERMES_CADUCEUS`).
    pub banner_hero: String,
}

impl Default for SkinConfig {
    fn default() -> Self {
        SkinConfig {
            name: String::new(),
            description: String::new(),
            colors: BTreeMap::new(),
            spinner: JsonValue::Object(serde_json::Map::new()),
            branding: BTreeMap::new(),
            tool_prefix: DEFAULT_TOOL_PREFIX.to_string(),
            tool_emojis: BTreeMap::new(),
            banner_logo: String::new(),
            banner_hero: String::new(),
        }
    }
}

impl SkinConfig {
    /// Get a color value with fallback.
    pub fn get_color(&self, key: &str, fallback: &str) -> String {
        self.colors
            .get(key)
            .cloned()
            .unwrap_or_else(|| fallback.to_string())
    }

    /// Get spinner wing pairs, or empty list if none. Each entry is a
    /// `[left, right]` pair; entries that are not 2-element lists are ignored.
    pub fn get_spinner_wings(&self) -> Vec<(String, String)> {
        let mut result = Vec::new();
        if let Some(raw) = self.spinner.get("wings").and_then(|w| w.as_array()) {
            for pair in raw {
                if let Some(arr) = pair.as_array() {
                    if arr.len() == 2 {
                        result.push((json_to_str(&arr[0]), json_to_str(&arr[1])));
                    }
                }
            }
        }
        result
    }

    /// Get a branding value with fallback.
    pub fn get_branding(&self, key: &str, fallback: &str) -> String {
        self.branding
            .get(key)
            .cloned()
            .unwrap_or_else(|| fallback.to_string())
    }
}

/// Stringify a JSON value the way Python's `str(...)` would for the leaf types
/// used in spinner wings (strings unquoted; everything else via JSON).
fn json_to_str(v: &JsonValue) -> String {
    match v {
        JsonValue::String(s) => s.clone(),
        JsonValue::Null => "None".to_string(),
        JsonValue::Bool(b) => {
            if *b {
                "True".to_string()
            } else {
                "False".to_string()
            }
        }
        other => other.to_string(),
    }
}

// =============================================================================
// Built-in skin definitions
// =============================================================================
//
// The raw built-in skins are embedded as a JSON document, which keeps the large
// ASCII-art logo/hero blocks readable and lets the loading path treat built-ins
// and user YAML uniformly.

/// Raw JSON for every built-in skin, keyed by name. Returned as a parsed
/// object so callers can introspect it.
pub fn builtin_skins() -> JsonValue {
    serde_json::from_str(BUILTIN_SKINS_JSON).expect("BUILTIN_SKINS_JSON must be valid JSON")
}

/// Names of all built-in skins, in definition order.
pub const BUILTIN_SKIN_NAMES: &[&str] = &[
    "default",
    "ares",
    "mono",
    "slate",
    "daylight",
    "warm-lightmode",
    "poseidon",
    "sisyphus",
    "charizard",
];

// NOTE: embedded inline (raw string) rather than an external data file so the
// module is fully self-contained. This is a 1:1 transcription of the Python
// `_BUILTIN_SKINS` table.
const BUILTIN_SKINS_JSON: &str = r##"{
  "default": {
    "name": "default",
    "description": "Classic Hermes — gold and kawaii",
    "colors": {
      "banner_border": "#CD7F32",
      "banner_title": "#FFD700",
      "banner_accent": "#FFBF00",
      "banner_dim": "#B8860B",
      "banner_text": "#FFF8DC",
      "ui_accent": "#FFBF00",
      "ui_label": "#DAA520",
      "ui_ok": "#4caf50",
      "ui_error": "#ef5350",
      "ui_warn": "#ffa726",
      "prompt": "#FFF8DC",
      "input_rule": "#CD7F32",
      "response_border": "#FFD700",
      "status_bar_bg": "#1a1a2e",
      "session_label": "#DAA520",
      "session_border": "#8B8682"
    },
    "spinner": {},
    "branding": {
      "agent_name": "Hermes Agent",
      "welcome": "Welcome to Hermes Agent! Type your message or /help for commands.",
      "goodbye": "Goodbye! ⚕",
      "response_label": " ⚕ Hermes ",
      "prompt_symbol": "❯",
      "help_header": "(^_^)? Available Commands"
    },
    "tool_prefix": "┊"
  },
  "ares": {
    "name": "ares",
    "description": "War-god theme — crimson and bronze",
    "colors": {
      "banner_border": "#9F1C1C",
      "banner_title": "#C7A96B",
      "banner_accent": "#DD4A3A",
      "banner_dim": "#6B1717",
      "banner_text": "#F1E6CF",
      "ui_accent": "#DD4A3A",
      "ui_label": "#C7A96B",
      "ui_ok": "#4caf50",
      "ui_error": "#ef5350",
      "ui_warn": "#ffa726",
      "prompt": "#F1E6CF",
      "input_rule": "#9F1C1C",
      "response_border": "#C7A96B",
      "status_bar_bg": "#2A1212",
      "status_bar_text": "#F1E6CF",
      "status_bar_strong": "#C7A96B",
      "status_bar_dim": "#6E584B",
      "status_bar_good": "#7BC96F",
      "status_bar_warn": "#C7A96B",
      "status_bar_bad": "#DD4A3A",
      "status_bar_critical": "#EF5350",
      "session_label": "#C7A96B",
      "session_border": "#6E584B"
    },
    "spinner": {
      "waiting_faces": ["(⚔)", "(⛨)", "(▲)", "(<>)", "(/)"],
      "thinking_faces": ["(⚔)", "(⛨)", "(▲)", "(⌁)", "(<>)"],
      "thinking_verbs": ["forging", "marching", "sizing the field", "holding the line", "hammering plans", "tempering steel", "plotting impact", "raising the shield"],
      "wings": [["⟪⚔", "⚔⟫"], ["⟪▲", "▲⟫"], ["⟪╸", "╺⟫"], ["⟪⛨", "⛨⟫"]]
    },
    "branding": {
      "agent_name": "Ares Agent",
      "welcome": "Welcome to Ares Agent! Type your message or /help for commands.",
      "goodbye": "Farewell, warrior! ⚔",
      "response_label": " ⚔ Ares ",
      "prompt_symbol": "⚔",
      "help_header": "(⚔) Available Commands"
    },
    "tool_prefix": "╎",
    "banner_logo": "[bold #A3261F] █████╗ ██████╗ ███████╗███████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/]\n[bold #B73122]██╔══██╗██╔══██╗██╔════╝██╔════╝      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/]\n[#C93C24]███████║██████╔╝█████╗  ███████╗█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║[/]\n[#D84A28]██╔══██║██╔══██╗██╔══╝  ╚════██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║[/]\n[#E15A2D]██║  ██║██║  ██║███████╗███████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║[/]\n[#EB6C32]╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]",
    "banner_hero": "[#9F1C1C]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣤⣤⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]\n[#9F1C1C]⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⠟⠻⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]\n[#C7A96B]⠀⠀⠀⠀⠀⠀⠀⣠⣾⡿⠋⠀⠀⠀⠙⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀[/]\n[#C7A96B]⠀⠀⠀⠀⠀⢀⣾⡿⠋⠀⠀⢠⡄⠀⠀⠙⢿⣷⡀⠀⠀⠀⠀⠀[/]\n[#DD4A3A]⠀⠀⠀⠀⣰⣿⠟⠀⠀⠀⣰⣿⣿⣆⠀⠀⠀⠻⣿⣆⠀⠀⠀⠀[/]\n[#DD4A3A]⠀⠀⠀⢰⣿⠏⠀⠀⢀⣾⡿⠉⢿⣷⡀⠀⠀⠹⣿⡆⠀⠀⠀[/]\n[#9F1C1C]⠀⠀⠀⣿⡟⠀⠀⣠⣿⠟⠀⠀⠀⠻⣿⣄⠀⠀⢻⣿⠀⠀⠀[/]\n[#9F1C1C]⠀⠀⠀⣿⡇⠀⠀⠙⠋⠀⠀⚔⠀⠀⠙⠋⠀⠀⢸⣿⠀⠀⠀[/]\n[#6B1717]⠀⠀⠀⢿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⡿⠀⠀⠀[/]\n[#6B1717]⠀⠀⠀⠘⢿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⡿⠃⠀⠀⠀[/]\n[#C7A96B]⠀⠀⠀⠀⠈⠻⣿⣷⣦⣤⣀⣀⣤⣤⣶⣿⠿⠋⠀⠀⠀⠀[/]\n[#C7A96B]⠀⠀⠀⠀⠀⠀⠀⠉⠛⠿⠿⠿⠿⠛⠉⠀⠀⠀⠀⠀⠀⠀[/]\n[#DD4A3A]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⚔⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]\n[dim #6B1717]⠀⠀⠀⠀⠀⠀⠀⠀war god online⠀⠀⠀⠀⠀⠀⠀⠀[/]"
  },
  "mono": {
    "name": "mono",
    "description": "Monochrome — clean grayscale",
    "colors": {
      "banner_border": "#555555",
      "banner_title": "#e6edf3",
      "banner_accent": "#aaaaaa",
      "banner_dim": "#444444",
      "banner_text": "#c9d1d9",
      "ui_accent": "#aaaaaa",
      "ui_label": "#888888",
      "ui_ok": "#888888",
      "ui_error": "#cccccc",
      "ui_warn": "#999999",
      "prompt": "#c9d1d9",
      "input_rule": "#444444",
      "response_border": "#aaaaaa",
      "status_bar_bg": "#1F1F1F",
      "status_bar_text": "#C9D1D9",
      "status_bar_strong": "#E6EDF3",
      "status_bar_dim": "#777777",
      "status_bar_good": "#B5B5B5",
      "status_bar_warn": "#AAAAAA",
      "status_bar_bad": "#D0D0D0",
      "status_bar_critical": "#F0F0F0",
      "session_label": "#888888",
      "session_border": "#555555"
    },
    "spinner": {},
    "branding": {
      "agent_name": "Hermes Agent",
      "welcome": "Welcome to Hermes Agent! Type your message or /help for commands.",
      "goodbye": "Goodbye! ⚕",
      "response_label": " ⚕ Hermes ",
      "prompt_symbol": "❯",
      "help_header": "[?] Available Commands"
    },
    "tool_prefix": "┊"
  },
  "slate": {
    "name": "slate",
    "description": "Cool blue — developer-focused",
    "colors": {
      "banner_border": "#4169e1",
      "banner_title": "#7eb8f6",
      "banner_accent": "#8EA8FF",
      "banner_dim": "#4b5563",
      "banner_text": "#c9d1d9",
      "ui_accent": "#7eb8f6",
      "ui_label": "#8EA8FF",
      "ui_ok": "#63D0A6",
      "ui_error": "#F7A072",
      "ui_warn": "#e6a855",
      "prompt": "#c9d1d9",
      "input_rule": "#4169e1",
      "response_border": "#7eb8f6",
      "status_bar_bg": "#151C2F",
      "status_bar_text": "#C9D1D9",
      "status_bar_strong": "#7EB8F6",
      "status_bar_dim": "#4B5563",
      "status_bar_good": "#63D0A6",
      "status_bar_warn": "#E6A855",
      "status_bar_bad": "#F7A072",
      "status_bar_critical": "#FF7A7A",
      "session_label": "#7eb8f6",
      "session_border": "#4b5563"
    },
    "spinner": {},
    "branding": {
      "agent_name": "Hermes Agent",
      "welcome": "Welcome to Hermes Agent! Type your message or /help for commands.",
      "goodbye": "Goodbye! ⚕",
      "response_label": " ⚕ Hermes ",
      "prompt_symbol": "❯",
      "help_header": "(^_^)? Available Commands"
    },
    "tool_prefix": "┊"
  },
  "daylight": {
    "name": "daylight",
    "description": "Light theme for bright terminals with dark text and cool blue accents",
    "colors": {
      "banner_border": "#2563EB",
      "banner_title": "#0F172A",
      "banner_accent": "#1D4ED8",
      "banner_dim": "#475569",
      "banner_text": "#111827",
      "ui_accent": "#2563EB",
      "ui_label": "#0F766E",
      "ui_ok": "#15803D",
      "ui_error": "#B91C1C",
      "ui_warn": "#B45309",
      "prompt": "#111827",
      "input_rule": "#93C5FD",
      "response_border": "#2563EB",
      "session_label": "#1D4ED8",
      "session_border": "#64748B",
      "status_bar_bg": "#E5EDF8",
      "voice_status_bg": "#E5EDF8",
      "completion_menu_bg": "#F8FAFC",
      "completion_menu_current_bg": "#DBEAFE",
      "completion_menu_meta_bg": "#EEF2FF",
      "completion_menu_meta_current_bg": "#BFDBFE"
    },
    "spinner": {},
    "branding": {
      "agent_name": "Hermes Agent",
      "welcome": "Welcome to Hermes Agent! Type your message or /help for commands.",
      "goodbye": "Goodbye! ⚕",
      "response_label": " ⚕ Hermes ",
      "prompt_symbol": "❯",
      "help_header": "[?] Available Commands"
    },
    "tool_prefix": "│"
  },
  "warm-lightmode": {
    "name": "warm-lightmode",
    "description": "Warm light mode — dark brown/gold text for light terminal backgrounds",
    "colors": {
      "banner_border": "#8B6914",
      "banner_title": "#5C3D11",
      "banner_accent": "#8B4513",
      "banner_dim": "#8B7355",
      "banner_text": "#2C1810",
      "ui_accent": "#8B4513",
      "ui_label": "#5C3D11",
      "ui_ok": "#2E7D32",
      "ui_error": "#C62828",
      "ui_warn": "#E65100",
      "prompt": "#2C1810",
      "input_rule": "#8B6914",
      "response_border": "#8B6914",
      "session_label": "#5C3D11",
      "session_border": "#A0845C",
      "status_bar_bg": "#F5F0E8",
      "voice_status_bg": "#F5F0E8",
      "completion_menu_bg": "#F5EFE0",
      "completion_menu_current_bg": "#E8DCC8",
      "completion_menu_meta_bg": "#F0E8D8",
      "completion_menu_meta_current_bg": "#DFCFB0"
    },
    "spinner": {},
    "branding": {
      "agent_name": "Hermes Agent",
      "welcome": "Welcome to Hermes Agent! Type your message or /help for commands.",
      "goodbye": "Goodbye! ⚕",
      "response_label": " ⚕ Hermes ",
      "prompt_symbol": "❯",
      "help_header": "(^_^)? Available Commands"
    },
    "tool_prefix": "┊"
  },
  "poseidon": {
    "name": "poseidon",
    "description": "Ocean-god theme — deep blue and seafoam",
    "colors": {
      "banner_border": "#2A6FB9",
      "banner_title": "#A9DFFF",
      "banner_accent": "#5DB8F5",
      "banner_dim": "#153C73",
      "banner_text": "#EAF7FF",
      "ui_accent": "#5DB8F5",
      "ui_label": "#A9DFFF",
      "ui_ok": "#4caf50",
      "ui_error": "#ef5350",
      "ui_warn": "#ffa726",
      "prompt": "#EAF7FF",
      "input_rule": "#2A6FB9",
      "response_border": "#5DB8F5",
      "status_bar_bg": "#0F2440",
      "status_bar_text": "#EAF7FF",
      "status_bar_strong": "#A9DFFF",
      "status_bar_dim": "#496884",
      "status_bar_good": "#6ED7B0",
      "status_bar_warn": "#5DB8F5",
      "status_bar_bad": "#2A6FB9",
      "status_bar_critical": "#D94F4F",
      "session_label": "#A9DFFF",
      "session_border": "#496884"
    },
    "spinner": {
      "waiting_faces": ["(≈)", "(Ψ)", "(∿)", "(◌)", "(◠)"],
      "thinking_faces": ["(Ψ)", "(∿)", "(≈)", "(⌁)", "(◌)"],
      "thinking_verbs": ["charting currents", "sounding the depth", "reading foam lines", "steering the trident", "tracking undertow", "plotting sea lanes", "calling the swell", "measuring pressure"],
      "wings": [["⟪≈", "≈⟫"], ["⟪Ψ", "Ψ⟫"], ["⟪∿", "∿⟫"], ["⟪◌", "◌⟫"]]
    },
    "branding": {
      "agent_name": "Poseidon Agent",
      "welcome": "Welcome to Poseidon Agent! Type your message or /help for commands.",
      "goodbye": "Fair winds! Ψ",
      "response_label": " Ψ Poseidon ",
      "prompt_symbol": "Ψ",
      "help_header": "(Ψ) Available Commands"
    },
    "tool_prefix": "│",
    "banner_logo": "[bold #B8E8FF]██████╗  ██████╗ ███████╗███████╗██╗██████╗  ██████╗ ███╗   ██╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/]\n[bold #97D6FF]██╔══██╗██╔═══██╗██╔════╝██╔════╝██║██╔══██╗██╔═══██╗████╗  ██║      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/]\n[#75C1F6]██████╔╝██║   ██║███████╗█████╗  ██║██║  ██║██║   ██║██╔██╗ ██║█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║[/]\n[#4FA2E0]██╔═══╝ ██║   ██║╚════██║██╔══╝  ██║██║  ██║██║   ██║██║╚██╗██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║[/]\n[#2E7CC7]██║     ╚██████╔╝███████║███████╗██║██████╔╝╚██████╔╝██║ ╚████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║[/]\n[#1B4F95]╚═╝      ╚═════╝ ╚══════╝╚══════╝╚═╝╚═════╝  ╚═════╝ ╚═╝  ╚═══╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]",
    "banner_hero": "[#2A6FB9]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]\n[#5DB8F5]⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]\n[#5DB8F5]⠀⠀⠀⠀⠀⠀⠀⢠⣿⠏⠀Ψ⠀⠹⣿⡄⠀⠀⠀⠀⠀⠀⠀[/]\n[#A9DFFF]⠀⠀⠀⠀⠀⠀⠀⣿⡟⠀⠀⠀⠀⠀⢻⣿⠀⠀⠀⠀⠀⠀⠀[/]\n[#A9DFFF]⠀⠀⠀≈≈≈≈≈⣿⡇⠀⠀⠀⠀⠀⢸⣿≈≈≈≈≈⠀⠀⠀[/]\n[#5DB8F5]⠀⠀⠀⠀⠀⠀⠀⣿⡇⠀⠀⠀⠀⠀⢸⣿⠀⠀⠀⠀⠀⠀⠀[/]\n[#2A6FB9]⠀⠀⠀⠀⠀⠀⠀⢿⣧⠀⠀⠀⠀⠀⣼⡿⠀⠀⠀⠀⠀⠀⠀[/]\n[#2A6FB9]⠀⠀⠀⠀⠀⠀⠀⠘⢿⣷⣄⣀⣠⣾⡿⠃⠀⠀⠀⠀⠀⠀⠀[/]\n[#153C73]⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣿⣿⡿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀[/]\n[#153C73]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]\n[#5DB8F5]⠀⠀⠀⠀⠀≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈⠀⠀⠀⠀⠀[/]\n[#A9DFFF]⠀⠀⠀⠀⠀⠀≈≈≈≈≈≈≈≈≈≈≈≈≈⠀⠀⠀⠀⠀⠀[/]\n[dim #153C73]⠀⠀⠀⠀⠀⠀⠀deep waters hold⠀⠀⠀⠀⠀⠀⠀[/]"
  },
  "sisyphus": {
    "name": "sisyphus",
    "description": "Sisyphean theme — austere grayscale with persistence",
    "colors": {
      "banner_border": "#B7B7B7",
      "banner_title": "#F5F5F5",
      "banner_accent": "#E7E7E7",
      "banner_dim": "#4A4A4A",
      "banner_text": "#D3D3D3",
      "ui_accent": "#E7E7E7",
      "ui_label": "#D3D3D3",
      "ui_ok": "#919191",
      "ui_error": "#E7E7E7",
      "ui_warn": "#B7B7B7",
      "prompt": "#F5F5F5",
      "input_rule": "#656565",
      "response_border": "#B7B7B7",
      "status_bar_bg": "#202020",
      "status_bar_text": "#D3D3D3",
      "status_bar_strong": "#F5F5F5",
      "status_bar_dim": "#656565",
      "status_bar_good": "#B7B7B7",
      "status_bar_warn": "#D3D3D3",
      "status_bar_bad": "#E7E7E7",
      "status_bar_critical": "#F5F5F5",
      "session_label": "#919191",
      "session_border": "#656565"
    },
    "spinner": {
      "waiting_faces": ["(◉)", "(◌)", "(◬)", "(⬤)", "(::)"],
      "thinking_faces": ["(◉)", "(◬)", "(◌)", "(○)", "(●)"],
      "thinking_verbs": ["finding traction", "measuring the grade", "resetting the boulder", "counting the ascent", "testing leverage", "setting the shoulder", "pushing uphill", "enduring the loop"],
      "wings": [["⟪◉", "◉⟫"], ["⟪◬", "◬⟫"], ["⟪◌", "◌⟫"], ["⟪⬤", "⬤⟫"]]
    },
    "branding": {
      "agent_name": "Sisyphus Agent",
      "welcome": "Welcome to Sisyphus Agent! Type your message or /help for commands.",
      "goodbye": "The boulder waits. ◉",
      "response_label": " ◉ Sisyphus ",
      "prompt_symbol": "◉",
      "help_header": "(◉) Available Commands"
    },
    "tool_prefix": "│",
    "banner_logo": "[bold #F5F5F5]███████╗██╗███████╗██╗   ██╗██████╗ ██╗  ██╗██╗   ██╗███████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/]\n[bold #E7E7E7]██╔════╝██║██╔════╝╚██╗ ██╔╝██╔══██╗██║  ██║██║   ██║██╔════╝      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/]\n[#D7D7D7]███████╗██║███████╗ ╚████╔╝ ██████╔╝███████║██║   ██║███████╗█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║[/]\n[#BFBFBF]╚════██║██║╚════██║  ╚██╔╝  ██╔═══╝ ██╔══██║██║   ██║╚════██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║[/]\n[#8F8F8F]███████║██║███████║   ██║   ██║     ██║  ██║╚██████╔╝███████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║[/]\n[#626262]╚══════╝╚═╝╚══════╝   ╚═╝   ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]",
    "banner_hero": "[#B7B7B7]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]\n[#D3D3D3]⠀⠀⠀⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀[/]\n[#E7E7E7]⠀⠀⠀⠀⠀⠀⣾⣿⣿⣿⣿⣿⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀[/]\n[#F5F5F5]⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀[/]\n[#E7E7E7]⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀[/]\n[#D3D3D3]⠀⠀⠀⠀⠀⠀⠘⢿⣿⣿⣿⣿⣿⡿⠃⠀⠀⠀⠀⠀⠀⠀[/]\n[#B7B7B7]⠀⠀⠀⠀⠀⠀⠀⠀⠙⠿⣿⠿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]\n[#919191]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]\n[#656565]⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]\n[#656565]⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]\n[#4A4A4A]⠀⠀⠀⠀⠀⠀⠀⣰⣿⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]\n[#4A4A4A]⠀⠀⠀⠀⠀⣀⣴⣿⣿⣿⣿⣿⣿⣦⣀⠀⠀⠀⠀⠀⠀[/]\n[#656565]⠀⠀⠀━━━━━━━━━━━━━━━━━━━━━━━⠀⠀⠀[/]\n[dim #4A4A4A]⠀⠀⠀⠀⠀⠀⠀⠀⠀the boulder⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]"
  },
  "charizard": {
    "name": "charizard",
    "description": "Volcanic theme — burnt orange and ember",
    "colors": {
      "banner_border": "#C75B1D",
      "banner_title": "#FFD39A",
      "banner_accent": "#F29C38",
      "banner_dim": "#7A3511",
      "banner_text": "#FFF0D4",
      "ui_accent": "#F29C38",
      "ui_label": "#FFD39A",
      "ui_ok": "#4caf50",
      "ui_error": "#ef5350",
      "ui_warn": "#ffa726",
      "prompt": "#FFF0D4",
      "input_rule": "#C75B1D",
      "response_border": "#F29C38",
      "status_bar_bg": "#2B160E",
      "status_bar_text": "#FFF0D4",
      "status_bar_strong": "#FFD39A",
      "status_bar_dim": "#6C4724",
      "status_bar_good": "#6BCB77",
      "status_bar_warn": "#F29C38",
      "status_bar_bad": "#E2832B",
      "status_bar_critical": "#EF5350",
      "session_label": "#FFD39A",
      "session_border": "#6C4724"
    },
    "spinner": {
      "waiting_faces": ["(✦)", "(▲)", "(◇)", "(<>)", "(🔥)"],
      "thinking_faces": ["(✦)", "(▲)", "(◇)", "(⌁)", "(🔥)"],
      "thinking_verbs": ["banking into the draft", "measuring burn", "reading the updraft", "tracking ember fall", "setting wing angle", "holding the flame core", "plotting a hot landing", "coiling for lift"],
      "wings": [["⟪✦", "✦⟫"], ["⟪▲", "▲⟫"], ["⟪◌", "◌⟫"], ["⟪◇", "◇⟫"]]
    },
    "branding": {
      "agent_name": "Charizard Agent",
      "welcome": "Welcome to Charizard Agent! Type your message or /help for commands.",
      "goodbye": "Flame out! ✦",
      "response_label": " ✦ Charizard ",
      "prompt_symbol": "✦",
      "help_header": "(✦) Available Commands"
    },
    "tool_prefix": "│",
    "banner_logo": "[bold #FFF0D4] ██████╗██╗  ██╗ █████╗ ██████╗ ██╗███████╗ █████╗ ██████╗ ██████╗        █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/]\n[bold #FFD39A]██╔════╝██║  ██║██╔══██╗██╔══██╗██║╚══███╔╝██╔══██╗██╔══██╗██╔══██╗      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/]\n[#F29C38]██║     ███████║███████║██████╔╝██║  ███╔╝ ███████║██████╔╝██║  ██║█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║[/]\n[#E2832B]██║     ██╔══██║██╔══██║██╔══██╗██║ ███╔╝  ██╔══██║██╔══██╗██║  ██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║[/]\n[#C75B1D]╚██████╗██║  ██║██║  ██║██║  ██║██║███████╗██║  ██║██║  ██║██████╔╝      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║[/]\n[#7A3511] ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝       ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]",
    "banner_hero": "[#FFD39A]⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⠶⠶⠶⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀[/]\n[#F29C38]⠀⠀⠀⠀⠀⠀⣴⠟⠁⠀⠀⠀⠀⠈⠻⣦⠀⠀⠀⠀⠀⠀[/]\n[#F29C38]⠀⠀⠀⠀⠀⣼⠏⠀⠀⠀✦⠀⠀⠀⠀⠹⣧⠀⠀⠀⠀⠀[/]\n[#E2832B]⠀⠀⠀⠀⢰⡟⠀⠀⣀⣤⣤⣤⣀⠀⠀⠀⢻⡆⠀⠀⠀⠀[/]\n[#E2832B]⠀⠀⣠⡾⠛⠁⣠⣾⠟⠉⠀⠉⠻⣷⣄⠀⠈⠛⢷⣄⠀⠀[/]\n[#C75B1D]⠀⣼⠟⠀⢀⣾⠟⠁⠀⠀⠀⠀⠀⠈⠻⣷⡀⠀⠻⣧⠀[/]\n[#C75B1D]⢸⡟⠀⠀⣿⡟⠀⠀⠀🔥⠀⠀⠀⠀⢻⣿⠀⠀⢻⡇[/]\n[#7A3511]⠀⠻⣦⡀⠘⢿⣧⡀⠀⠀⠀⠀⠀⢀⣼⡿⠃⢀⣴⠟⠀[/]\n[#7A3511]⠀⠀⠈⠻⣦⣀⠙⢿⣷⣤⣤⣤⣾⡿⠋⣀⣴⠟⠁⠀⠀[/]\n[#C75B1D]⠀⠀⠀⠀⠈⠙⠛⠶⠤⠭⠭⠤⠶⠛⠋⠁⠀⠀⠀⠀[/]\n[#F29C38]⠀⠀⠀⠀⠀⠀⠀⠀⣰⡿⢿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]\n[#F29C38]⠀⠀⠀⠀⠀⠀⠀⣼⡟⠀⠀⢻⣧⠀⠀⠀⠀⠀⠀⠀⠀[/]\n[dim #7A3511]⠀⠀⠀⠀⠀⠀⠀tail flame lit⠀⠀⠀⠀⠀⠀⠀⠀[/]"
  }
}"##;

// =============================================================================
// Skin loading and management
// =============================================================================

struct ActiveState {
    skin: Option<SkinConfig>,
    name: String,
}

fn active_state() -> &'static Mutex<ActiveState> {
    use std::sync::OnceLock;
    static STATE: OnceLock<Mutex<ActiveState>> = OnceLock::new();
    STATE.get_or_init(|| {
        Mutex::new(ActiveState {
            skin: None,
            name: "default".to_string(),
        })
    })
}

/// User skins directory: `<hermes_home>/skins`.
pub fn skins_dir() -> PathBuf {
    hermes_home().join("skins")
}

/// Resolve the Hermes home directory. Mirrors `hermes_constants.get_hermes_home`
/// for the common case (the `HERMES_HOME` env override, else `~/.hermes`).
fn hermes_home() -> PathBuf {
    if let Ok(val) = std::env::var("HERMES_HOME") {
        let trimmed = val.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".hermes")
}

/// Load a skin definition from a YAML file. Returns `None` on any error or if
/// the document is not a mapping containing a `name` key.
pub fn load_skin_from_yaml(path: &std::path::Path) -> Option<YamlValue> {
    let text = std::fs::read_to_string(path).ok()?;
    let data: YamlValue = serde_yaml::from_str(&text).ok()?;
    if let YamlValue::Mapping(ref map) = data {
        if map.contains_key(YamlValue::String("name".to_string())) {
            return Some(data);
        }
    }
    None
}

/// Extract a string field from a generic mapping (YAML or JSON-backed YAML).
fn yaml_get_str(map: &serde_yaml::Mapping, key: &str) -> Option<String> {
    map.get(YamlValue::String(key.to_string())).and_then(|v| {
        v.as_str().map(|s| s.to_string())
    })
}

/// Extract a string->string mapping field (colors/branding/tool_emojis).
fn yaml_get_str_map(map: &serde_yaml::Mapping, key: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    if let Some(YamlValue::Mapping(inner)) = map.get(YamlValue::String(key.to_string())) {
        for (k, v) in inner {
            if let (Some(ks), Some(vs)) = (k.as_str(), v.as_str()) {
                out.insert(ks.to_string(), vs.to_string());
            }
        }
    }
    out
}

/// Convert a YAML value into a JSON value (for the spinner blob).
fn yaml_to_json(v: &YamlValue) -> JsonValue {
    match v {
        YamlValue::Null => JsonValue::Null,
        YamlValue::Bool(b) => JsonValue::Bool(*b),
        YamlValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                JsonValue::Number(i.into())
            } else if let Some(u) = n.as_u64() {
                JsonValue::Number(u.into())
            } else if let Some(f) = n.as_f64() {
                serde_json::Number::from_f64(f)
                    .map(JsonValue::Number)
                    .unwrap_or(JsonValue::Null)
            } else {
                JsonValue::Null
            }
        }
        YamlValue::String(s) => JsonValue::String(s.clone()),
        YamlValue::Sequence(seq) => {
            JsonValue::Array(seq.iter().map(yaml_to_json).collect())
        }
        YamlValue::Mapping(map) => {
            let mut obj = serde_json::Map::new();
            for (k, val) in map {
                let key = match k {
                    YamlValue::String(s) => s.clone(),
                    other => json_to_str(&yaml_to_json(other)),
                };
                obj.insert(key, yaml_to_json(val));
            }
            JsonValue::Object(obj)
        }
        YamlValue::Tagged(t) => yaml_to_json(&t.value),
    }
}

/// Build a `SkinConfig` from a raw mapping (built-in or loaded from YAML).
/// Missing color/spinner/branding keys inherit from the `default` skin.
fn build_skin_config(data: &serde_yaml::Mapping) -> SkinConfig {
    // The `default` skin, as a YAML mapping, supplies the inherited base.
    let builtins = builtin_skins();
    let default = builtins
        .get("default")
        .cloned()
        .unwrap_or(JsonValue::Object(serde_json::Map::new()));
    let default_yaml = json_value_to_yaml(&default);
    let default_map = match default_yaml {
        YamlValue::Mapping(m) => m,
        _ => serde_yaml::Mapping::new(),
    };

    // Colors: default colors, overridden by data colors.
    let mut colors = yaml_get_str_map(&default_map, "colors");
    colors.extend(yaml_get_str_map(data, "colors"));

    // Spinner: default spinner, updated by data spinner.
    let mut spinner_obj = serde_json::Map::new();
    if let Some(d) = default_map.get(YamlValue::String("spinner".to_string())) {
        if let JsonValue::Object(m) = yaml_to_json(d) {
            spinner_obj = m;
        }
    }
    if let Some(d) = data.get(YamlValue::String("spinner".to_string())) {
        if let JsonValue::Object(m) = yaml_to_json(d) {
            for (k, v) in m {
                spinner_obj.insert(k, v);
            }
        }
    }

    // Branding: default branding, overridden by data branding.
    let mut branding = yaml_get_str_map(&default_map, "branding");
    branding.extend(yaml_get_str_map(data, "branding"));

    let name = yaml_get_str(data, "name").unwrap_or_else(|| "unknown".to_string());
    let description = yaml_get_str(data, "description").unwrap_or_default();
    let tool_prefix = yaml_get_str(data, "tool_prefix")
        .or_else(|| yaml_get_str(&default_map, "tool_prefix"))
        .unwrap_or_else(|| DEFAULT_TOOL_PREFIX.to_string());
    let tool_emojis = yaml_get_str_map(data, "tool_emojis");
    let banner_logo = yaml_get_str(data, "banner_logo").unwrap_or_default();
    let banner_hero = yaml_get_str(data, "banner_hero").unwrap_or_default();

    SkinConfig {
        name,
        description,
        colors,
        spinner: JsonValue::Object(spinner_obj),
        branding,
        tool_prefix,
        tool_emojis,
        banner_logo,
        banner_hero,
    }
}

/// Convert a JSON value to a YAML value (used to feed JSON built-ins through the
/// same mapping-based code paths as user YAML).
fn json_value_to_yaml(v: &JsonValue) -> YamlValue {
    match v {
        JsonValue::Null => YamlValue::Null,
        JsonValue::Bool(b) => YamlValue::Bool(*b),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                YamlValue::Number(i.into())
            } else if let Some(u) = n.as_u64() {
                YamlValue::Number(u.into())
            } else if let Some(f) = n.as_f64() {
                YamlValue::Number(f.into())
            } else {
                YamlValue::Null
            }
        }
        JsonValue::String(s) => YamlValue::String(s.clone()),
        JsonValue::Array(arr) => {
            YamlValue::Sequence(arr.iter().map(json_value_to_yaml).collect())
        }
        JsonValue::Object(obj) => {
            let mut map = serde_yaml::Mapping::new();
            for (k, val) in obj {
                map.insert(YamlValue::String(k.clone()), json_value_to_yaml(val));
            }
            YamlValue::Mapping(map)
        }
    }
}

/// Summary of an available skin for listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkinSummary {
    pub name: String,
    pub description: String,
    /// `"builtin"` or `"user"`.
    pub source: String,
}

/// List all available skins (built-in + user-installed).
pub fn list_skins() -> Vec<SkinSummary> {
    let mut result: Vec<SkinSummary> = Vec::new();

    let builtins = builtin_skins();
    for name in BUILTIN_SKIN_NAMES {
        let desc = builtins
            .get(*name)
            .and_then(|s| s.get("description"))
            .and_then(|d| d.as_str())
            .unwrap_or("")
            .to_string();
        result.push(SkinSummary {
            name: name.to_string(),
            description: desc,
            source: "builtin".to_string(),
        });
    }

    let skins_path = skins_dir();
    if skins_path.is_dir() {
        let mut files: Vec<PathBuf> = match std::fs::read_dir(&skins_path) {
            Ok(entries) => entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("yaml"))
                .collect(),
            Err(_) => Vec::new(),
        };
        files.sort();

        for f in files {
            if let Some(data) = load_skin_from_yaml(&f) {
                if let YamlValue::Mapping(ref map) = data {
                    let stem = f
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_string();
                    let skin_name = yaml_get_str(map, "name").unwrap_or(stem);
                    // Skip if it shadows a built-in (or an earlier user skin).
                    if result.iter().any(|s| s.name == skin_name) {
                        continue;
                    }
                    let description = yaml_get_str(map, "description").unwrap_or_default();
                    result.push(SkinSummary {
                        name: skin_name,
                        description,
                        source: "user".to_string(),
                    });
                }
            }
        }
    }

    result
}

/// Load a skin by name. Checks user skins first, then built-in, then falls back
/// to `default`.
pub fn load_skin(name: &str) -> SkinConfig {
    // Check user skins directory.
    let user_file = skins_dir().join(format!("{name}.yaml"));
    if user_file.is_file() {
        if let Some(YamlValue::Mapping(map)) = load_skin_from_yaml(&user_file) {
            return build_skin_config(&map);
        }
    }

    // Check built-in skins.
    let builtins = builtin_skins();
    if let Some(data) = builtins.get(name) {
        if let YamlValue::Mapping(map) = json_value_to_yaml(data) {
            return build_skin_config(&map);
        }
    }

    // Fallback to default.
    log::warn!("Skin '{name}' not found, using default");
    if let Some(data) = builtins.get("default") {
        if let YamlValue::Mapping(map) = json_value_to_yaml(data) {
            return build_skin_config(&map);
        }
    }
    SkinConfig::default()
}

/// Get the currently active skin config (cached).
pub fn get_active_skin() -> SkinConfig {
    let mut state = active_state().lock().unwrap();
    if state.skin.is_none() {
        let name = state.name.clone();
        state.skin = Some(load_skin(&name));
    }
    state.skin.clone().unwrap()
}

/// Switch the active skin. Returns the new `SkinConfig`.
pub fn set_active_skin(name: &str) -> SkinConfig {
    let loaded = load_skin(name);
    let mut state = active_state().lock().unwrap();
    state.name = name.to_string();
    state.skin = Some(loaded.clone());
    loaded
}

/// Get the name of the currently active skin.
pub fn get_active_skin_name() -> String {
    active_state().lock().unwrap().name.clone()
}

/// Initialize the active skin from CLI config at startup. Call this once during
/// CLI init with the loaded config (as a JSON object).
pub fn init_skin_from_config(config: &JsonValue) {
    let display = config.get("display");
    let skin_name = display
        .and_then(|d| d.as_object())
        .and_then(|d| d.get("skin"))
        .and_then(|s| s.as_str());

    match skin_name {
        Some(s) if !s.trim().is_empty() => {
            set_active_skin(s.trim());
        }
        _ => {
            set_active_skin("default");
        }
    }
}

// =============================================================================
// Convenience helpers for CLI modules
// =============================================================================

/// Return the interactive prompt symbol with a single trailing space.
///
/// Skins store `prompt_symbol` as a bare token (no spaces). The trailing space
/// is appended here so callers can drop it straight into a rendered prompt.
pub fn get_active_prompt_symbol(fallback: &str) -> String {
    let raw = get_active_skin().get_branding("prompt_symbol", fallback);
    let raw = if raw.is_empty() {
        fallback.to_string()
    } else {
        raw
    };
    let cleaned = raw.trim();
    let cleaned = if cleaned.is_empty() {
        fallback.trim()
    } else {
        cleaned
    };
    format!("{cleaned} ")
}

/// Get the `/help` header from the active skin.
pub fn get_active_help_header(fallback: &str) -> String {
    get_active_skin().get_branding("help_header", fallback)
}

/// Get the goodbye line from the active skin.
pub fn get_active_goodbye(fallback: &str) -> String {
    get_active_skin().get_branding("goodbye", fallback)
}

/// Return `prompt_toolkit` style overrides derived from the active skin.
///
/// These are layered on top of the CLI's base TUI style so `/skin` can refresh
/// the live prompt_toolkit UI immediately without rebuilding the app.
pub fn get_prompt_toolkit_style_overrides() -> BTreeMap<String, String> {
    let skin = get_active_skin();

    let prompt = skin.get_color("prompt", "#FFF8DC");
    let input_rule = skin.get_color("input_rule", "#CD7F32");
    let title = skin.get_color("banner_title", "#FFD700");
    let text = skin.get_color("banner_text", &prompt);
    let dim = skin.get_color("banner_dim", "#555555");
    let label = skin.get_color("ui_label", &title);
    let warn = skin.get_color("ui_warn", "#FF8C00");
    let error = skin.get_color("ui_error", "#FF6B6B");
    let status_bg = skin.get_color("status_bar_bg", "#1a1a2e");
    let status_text = skin.get_color("status_bar_text", &text);
    let status_strong = skin.get_color("status_bar_strong", &title);
    let status_dim = skin.get_color("status_bar_dim", &dim);
    let ui_ok = skin.get_color("ui_ok", "#8FBC8F");
    let status_good = skin.get_color("status_bar_good", &ui_ok);
    let status_warn = skin.get_color("status_bar_warn", &warn);
    let banner_accent = skin.get_color("banner_accent", &warn);
    let status_bad = skin.get_color("status_bar_bad", &banner_accent);
    let status_critical = skin.get_color("status_bar_critical", &error);
    let voice_bg = skin.get_color("voice_status_bg", &status_bg);
    let menu_bg = skin.get_color("completion_menu_bg", "#1a1a2e");
    let menu_current_bg = skin.get_color("completion_menu_current_bg", "#333355");
    let menu_meta_bg = skin.get_color("completion_menu_meta_bg", &menu_bg);
    let menu_meta_current_bg =
        skin.get_color("completion_menu_meta_current_bg", &menu_current_bg);

    let mut out: BTreeMap<String, String> = BTreeMap::new();
    out.insert("input-area".into(), prompt.clone());
    out.insert("placeholder".into(), format!("{dim} italic"));
    out.insert("prompt".into(), prompt.clone());
    out.insert("prompt-working".into(), format!("{dim} italic"));
    out.insert("hint".into(), format!("{dim} italic"));
    out.insert("status-bar".into(), format!("bg:{status_bg} {status_text}"));
    out.insert(
        "status-bar-strong".into(),
        format!("bg:{status_bg} {status_strong} bold"),
    );
    out.insert(
        "status-bar-dim".into(),
        format!("bg:{status_bg} {status_dim}"),
    );
    out.insert(
        "status-bar-good".into(),
        format!("bg:{status_bg} {status_good} bold"),
    );
    out.insert(
        "status-bar-warn".into(),
        format!("bg:{status_bg} {status_warn} bold"),
    );
    out.insert(
        "status-bar-bad".into(),
        format!("bg:{status_bg} {status_bad} bold"),
    );
    out.insert(
        "status-bar-critical".into(),
        format!("bg:{status_bg} {status_critical} bold"),
    );
    out.insert("input-rule".into(), input_rule.clone());
    out.insert("image-badge".into(), format!("{label} bold"));
    out.insert(
        "completion-menu".into(),
        format!("bg:{menu_bg} {text}"),
    );
    out.insert(
        "completion-menu.completion".into(),
        format!("bg:{menu_bg} {text}"),
    );
    out.insert(
        "completion-menu.completion.current".into(),
        format!("bg:{menu_current_bg} {title}"),
    );
    out.insert(
        "completion-menu.meta.completion".into(),
        format!("bg:{menu_meta_bg} {dim}"),
    );
    out.insert(
        "completion-menu.meta.completion.current".into(),
        format!("bg:{menu_meta_current_bg} {label}"),
    );
    out.insert("clarify-border".into(), input_rule.clone());
    out.insert("clarify-title".into(), format!("{title} bold"));
    out.insert("clarify-question".into(), format!("{text} bold"));
    out.insert("clarify-choice".into(), dim.clone());
    out.insert("clarify-selected".into(), format!("{title} bold"));
    out.insert("clarify-active-other".into(), format!("{title} italic"));
    out.insert("clarify-countdown".into(), input_rule.clone());
    out.insert("sudo-prompt".into(), format!("{error} bold"));
    out.insert("sudo-border".into(), input_rule.clone());
    out.insert("sudo-title".into(), format!("{error} bold"));
    out.insert("sudo-text".into(), text.clone());
    out.insert("approval-border".into(), input_rule.clone());
    out.insert("approval-title".into(), format!("{warn} bold"));
    out.insert("approval-desc".into(), format!("{text} bold"));
    out.insert("approval-cmd".into(), format!("{dim} italic"));
    out.insert("approval-choice".into(), dim.clone());
    out.insert("approval-selected".into(), format!("{title} bold"));
    out.insert("voice-status".into(), format!("bg:{voice_bg} {label}"));
    out.insert(
        "voice-status-recording".into(),
        format!("bg:{voice_bg} {error} bold"),
    );

    out
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_parse() {
        let b = builtin_skins();
        assert!(b.is_object());
        for name in BUILTIN_SKIN_NAMES {
            assert!(b.get(*name).is_some(), "missing builtin: {name}");
        }
    }

    #[test]
    fn default_skin_colors() {
        let skin = load_skin("default");
        assert_eq!(skin.name, "default");
        assert_eq!(skin.get_color("banner_title", ""), "#FFD700");
        assert_eq!(skin.get_color("banner_border", ""), "#CD7F32");
        assert_eq!(skin.tool_prefix, "┊");
        assert_eq!(skin.get_branding("agent_name", ""), "Hermes Agent");
        assert_eq!(skin.get_branding("prompt_symbol", ""), "❯");
    }

    #[test]
    fn ares_inherits_and_overrides() {
        let skin = load_skin("ares");
        assert_eq!(skin.name, "ares");
        // Overridden color.
        assert_eq!(skin.get_color("banner_border", ""), "#9F1C1C");
        // Branding override.
        assert_eq!(skin.get_branding("agent_name", ""), "Ares Agent");
        // tool_prefix override.
        assert_eq!(skin.tool_prefix, "╎");
        // banner_logo present.
        assert!(skin.banner_logo.contains("█"));
        // Spinner wings present and well-formed.
        let wings = skin.get_spinner_wings();
        assert_eq!(wings.len(), 4);
        assert_eq!(wings[0], ("⟪⚔".to_string(), "⚔⟫".to_string()));
    }

    #[test]
    fn unknown_skin_falls_back_to_default() {
        let skin = load_skin("does-not-exist-xyz");
        // Falls back to default content (name reflects the default skin).
        assert_eq!(skin.name, "default");
        assert_eq!(skin.get_color("banner_title", ""), "#FFD700");
    }

    #[test]
    fn prompt_symbol_appends_space() {
        set_active_skin("default");
        let sym = get_active_prompt_symbol("X");
        assert_eq!(sym, "❯ ");
    }

    #[test]
    fn prompt_symbol_fallback_when_blank() {
        // ares prompt symbol is "⚔"
        set_active_skin("ares");
        let sym = get_active_prompt_symbol("X");
        assert_eq!(sym, "⚔ ");
        set_active_skin("default");
    }

    #[test]
    fn set_and_get_active_skin() {
        set_active_skin("mono");
        assert_eq!(get_active_skin_name(), "mono");
        let skin = get_active_skin();
        assert_eq!(skin.name, "mono");
        set_active_skin("default");
    }

    #[test]
    fn list_skins_includes_builtins() {
        let skins = list_skins();
        let names: Vec<&str> = skins.iter().map(|s| s.name.as_str()).collect();
        for n in BUILTIN_SKIN_NAMES {
            assert!(names.contains(n), "missing {n} in listing");
        }
        assert!(skins.iter().all(|s| s.source == "builtin" || s.source == "user"));
        assert!(skins.iter().any(|s| s.name == "default" && s.source == "builtin"));
    }

    #[test]
    fn init_skin_from_config_reads_display() {
        let cfg = serde_json::json!({"display": {"skin": "slate"}});
        init_skin_from_config(&cfg);
        assert_eq!(get_active_skin_name(), "slate");

        let cfg2 = serde_json::json!({"display": {"skin": "  "}});
        init_skin_from_config(&cfg2);
        assert_eq!(get_active_skin_name(), "default");

        let cfg3 = serde_json::json!({});
        init_skin_from_config(&cfg3);
        assert_eq!(get_active_skin_name(), "default");
    }

    #[test]
    fn style_overrides_have_expected_keys() {
        set_active_skin("default");
        let styles = get_prompt_toolkit_style_overrides();
        assert_eq!(styles.get("input-area").map(|s| s.as_str()), Some("#FFF8DC"));
        assert_eq!(styles.get("prompt").map(|s| s.as_str()), Some("#FFF8DC"));
        assert_eq!(
            styles.get("status-bar").map(|s| s.as_str()),
            Some("bg:#1a1a2e #FFF8DC")
        );
        assert!(styles.contains_key("voice-status-recording"));
        set_active_skin("default");
    }
}
