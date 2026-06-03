//! Process-wide voice recording + TTS API for the TUI gateway.
//!
//! Faithful native port of `hermes_cli/voice.py`. Wraps an audio recorder
//! (recording/transcription) and a text-to-speech backend behind idempotent,
//! stateful entry points that the gateway's `voice.record`, `voice.toggle`,
//! and `voice.tts` JSON-RPC handlers can call from a dedicated thread.
//!
//! Two usage modes are exposed:
//!
//! * **Push-to-talk** ([`start_recording`] / [`stop_and_transcribe`]) — single
//!   manually-bounded capture used when the caller drives the start/stop pair
//!   explicitly.
//! * **Continuous (VAD)** ([`start_continuous`] / [`stop_continuous`]) — mirrors
//!   the classic CLI voice mode: recording auto-stops on silence, transcribes,
//!   hands the result to a callback, and then auto-restarts for the next turn.
//!   Three consecutive no-speech cycles stop the loop and fire
//!   `on_silent_limit` so the UI can turn the mode off.
//!
//! Because the Python module leans on optional audio deps (`sounddevice`,
//! `faster-whisper`, `numpy`), the hardware-facing surface is abstracted behind
//! the [`AudioRecorder`], [`Transcriber`], [`Beeper`], and [`TtsBackend`]
//! traits. The pure config-parsing helpers
//! ([`normalize_voice_record_key_for_prompt_toolkit`],
//! [`format_voice_record_key_for_status`], [`voice_record_key_from_config`])
//! are fully native and unit-tested.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use serde_json::Value;

// ── Modifier / key alias tables (mirrored from ui-tui/src/lib/platform.ts) ──

/// `ctrl`/`control` → `c-`, `alt`/`option`/`opt` → `a-`.
fn voice_mod_alias(token: &str) -> Option<&'static str> {
    match token {
        "ctrl" | "control" => Some("c-"),
        "alt" | "option" | "opt" => Some("a-"),
        _ => None,
    }
}

/// Named keys prompt_toolkit accepts in `c-<name>` / `a-<name>` form. Aliases
/// collapse to prompt_toolkit's canonical spelling so the same config value
/// binds identically in both runtimes.
fn voice_named_key(token: &str) -> Option<&'static str> {
    match token {
        "space" | "spc" => Some("space"),
        "enter" | "return" | "ret" => Some("enter"),
        "tab" => Some("tab"),
        "escape" | "esc" => Some("escape"),
        "backspace" | "bs" => Some("backspace"),
        "delete" | "del" => Some("delete"),
        _ => None,
    }
}

/// `useInputHandlers()` intercepts these before the voice check runs, so a
/// binding like `ctrl+c` (interrupt), `ctrl+d` (quit), or `ctrl+l` (clear
/// screen) would never fire push-to-talk — the same blocklist the TUI parser
/// uses.
fn is_reserved_ctrl_char(c: &str) -> bool {
    matches!(c, "c" | "d" | "l")
}

/// On macOS the classic CLI's prompt_toolkit bindings for copy / exit / clear
/// also claim `a-c` / `a-d` / `a-l`. Mirror the TUI parser's darwin-only
/// reservation so `option+c` etc. don't bind Alt+C in the CLI.
fn is_reserved_alt_char_mac(c: &str) -> bool {
    matches!(c, "c" | "d" | "l")
}

/// Documented default prompt_toolkit binding (Ctrl+B).
pub const DEFAULT_PT_KEY: &str = "c-b";

/// `true` on macOS. Extracted so tests can exercise both branches.
fn is_darwin() -> bool {
    cfg!(target_os = "macos")
}

/// Shape-safe `cfg.voice.record_key` lookup.
///
/// `load_config()` deep-merges raw YAML and preserves scalar overrides, so a
/// hand-edited `voice: true` / `voice: cmd+b` leaves `cfg["voice"]` as a
/// bool/str instead of a map. Returns `None` for malformed shapes so call
/// sites can feed the result straight into the normalizer/formatter and get
/// the documented default.
pub fn voice_record_key_from_config(cfg: &Value) -> Option<Value> {
    let obj = cfg.as_object()?;
    let voice = obj.get("voice")?;
    let voice_obj = voice.as_object()?;
    voice_obj.get("record_key").cloned()
}

/// Coerce `voice.record_key` into prompt_toolkit's `c-x` / `a-x` format.
///
/// Mirrors the TUI parser contract so one config value binds the same shortcut
/// in both runtimes:
///
/// * non-string / empty / typo'd / bare-char / multi-modifier / reserved
///   `ctrl+c|d|l` → documented default `c-b`
/// * single-char keys: `ctrl+o` → `c-o`
/// * named keys: `ctrl+space` → `c-space` (aliases collapse: `ctrl+return` →
///   `c-enter`)
/// * `super` / `win` / `windows` → `c-b` (TUI-only modifiers)
pub fn normalize_voice_record_key_for_prompt_toolkit(raw: &Value) -> String {
    let raw_str = match raw.as_str() {
        Some(s) => s,
        None => return DEFAULT_PT_KEY.to_string(),
    };

    let lowered = raw_str.trim().to_lowercase();
    if lowered.is_empty() {
        return DEFAULT_PT_KEY.to_string();
    }

    let parts: Vec<&str> = lowered
        .split('+')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect();
    if parts.is_empty() {
        return DEFAULT_PT_KEY.to_string();
    }

    // Multi-modifier chords like `ctrl+alt+r` bind different shortcuts in
    // prompt_toolkit and hermes-ink rejects them; collapse to the default.
    if parts.len() > 2 {
        return DEFAULT_PT_KEY.to_string();
    }

    // Bare char / bare named key (no explicit modifier) — rejected so both
    // runtimes agree.
    if parts.len() == 1 {
        return DEFAULT_PT_KEY.to_string();
    }

    let modifier_token = parts[0];
    let key_token = parts[1];

    // `super` / `win` / `windows` are TUI-only (prompt_toolkit has no super
    // modifier). Fall back to the documented default.
    if matches!(modifier_token, "super" | "win" | "windows") {
        return DEFAULT_PT_KEY.to_string();
    }

    let normalized_mod = match voice_mod_alias(modifier_token) {
        Some(m) => m,
        None => return DEFAULT_PT_KEY.to_string(),
    };

    // Single-char key: reject reserved-ctrl chords plus the mac-only alt
    // reservation.
    if key_token.chars().count() == 1 {
        if normalized_mod == "c-" && is_reserved_ctrl_char(key_token) {
            return DEFAULT_PT_KEY.to_string();
        }
        if normalized_mod == "a-" && is_darwin() && is_reserved_alt_char_mac(key_token) {
            return DEFAULT_PT_KEY.to_string();
        }
        return format!("{normalized_mod}{key_token}");
    }

    // Multi-char key token must be a known named key; typos like `ctrl+spcae`
    // fall back to the default rather than being passed through.
    match voice_named_key(key_token) {
        Some(named) => format!("{normalized_mod}{named}"),
        None => DEFAULT_PT_KEY.to_string(),
    }
}

/// Render `voice.record_key` for `/voice status` in CLI-friendly form.
///
/// Mirrors the TUI's `formatVoiceRecordKey`: returns `Ctrl+B` / `Alt+Space` /
/// `Ctrl+Enter`. Malformed configs surface as the documented default.
pub fn format_voice_record_key_for_status(raw: &Value) -> String {
    let normalized = normalize_voice_record_key_for_prompt_toolkit(raw);

    let (prefix, key): (String, String) = if let Some(rest) = normalized.strip_prefix("c-") {
        ("Ctrl+".to_string(), rest.to_string())
    } else if let Some(rest) = normalized.strip_prefix("a-") {
        ("Alt+".to_string(), rest.to_string())
    } else if let Some(idx) = normalized.find('+') {
        // `super+<key>` / `win+<key>` — CLI won't bind them, but render in
        // title case so status output is still readable.
        let modr = &normalized[..idx];
        let key = &normalized[idx + 1..];
        let mut chars = modr.chars();
        let title = match chars.next() {
            Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            None => String::new(),
        };
        (format!("{title}+"), key.to_string())
    } else {
        return "Ctrl+B".to_string();
    };

    if key.is_empty() {
        return prefix.trim_end_matches('+').to_string();
    }

    if key.chars().count() == 1 {
        return prefix + &key.to_uppercase();
    }

    let mut chars = key.chars();
    let titled = match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    };
    prefix + &titled
}

// ── Hardware-facing abstractions ─────────────────────────────────────────
//
// The Python module imports `create_audio_recorder`, `transcribe_recording`,
// `is_whisper_hallucination`, `play_audio_file`, `play_beep`, and
// `text_to_speech_tool` from `tools.voice_mode` / `tools.tts_tool`. Those touch
// real audio hardware and optional native deps, so the stateful loop logic is
// ported against these traits. A caller wires concrete implementations in;
// tests use fakes.

/// A microphone capture device. Mirrors `tools.voice_mode.AudioRecorder`.
pub trait AudioRecorder: Send {
    /// Whether a recording is currently in progress.
    fn is_recording(&self) -> bool;

    /// Apply the VAD silence threshold (RMS) for the next/active capture.
    fn set_silence_threshold(&mut self, threshold: i64);

    /// Apply the VAD silence duration (seconds) for the next/active capture.
    fn set_silence_duration(&mut self, duration: f64);

    /// Begin capturing. When `on_silence_stop` is `Some`, the recorder fires
    /// the callback (in a background thread) once VAD detects sustained
    /// silence. May return an error if the audio stream cannot be opened.
    fn start(&mut self, on_silence_stop: Option<SilenceCallback>) -> Result<(), String>;

    /// Stop the active recording and return the path to a written WAV file, or
    /// `None` when no speech was captured.
    fn stop(&mut self) -> Option<String>;

    /// Discard buffered frames without transcribing.
    fn cancel(&mut self);

    /// Peak RMS observed during the last capture (`-1` if unavailable).
    fn peak_rms(&self) -> i64 {
        -1
    }
}

/// Daemon-thread callback fired by [`AudioRecorder`] on sustained silence.
pub type SilenceCallback = Arc<dyn Fn() + Send + Sync>;

/// Transcribes a captured WAV file. Mirrors
/// `tools.voice_mode.transcribe_recording`.
pub trait Transcriber: Send + Sync {
    /// Returns `{"success": bool, "transcript": str, "error": str?}` exactly
    /// like the Python tool.
    fn transcribe(&self, wav_path: &str) -> Result<Value, String>;

    /// Whether the transcript is a known Whisper hallucination.
    fn is_hallucination(&self, text: &str) -> bool;
}

/// Plays start/stop beep cues. Mirrors `tools.voice_mode.play_beep`.
pub trait Beeper: Send + Sync {
    fn play_beep(&self, frequency: i32, count: i32);
}

/// Synthesizes + plays TTS. Mirrors `tools.tts_tool.text_to_speech_tool` plus
/// `tools.voice_mode.play_audio_file`.
pub trait TtsBackend: Send + Sync {
    /// Synthesize `text` to `output_path` (an mp3 path).
    fn text_to_speech(&self, text: &str, output_path: &str) -> Result<(), String>;

    /// Play a previously written audio file.
    fn play_audio_file(&self, path: &str);

    /// Whether beep cues are enabled (CLI parity: `voice.beep_enabled`,
    /// default `true`).
    fn beeps_enabled(&self) -> bool {
        true
    }
}

const CONTINUOUS_NO_SPEECH_LIMIT: i32 = 3;

/// Optional UI callbacks for the continuous loop.
#[derive(Clone, Default)]
pub struct ContinuousCallbacks {
    /// Called with each successfully transcribed turn.
    pub on_transcript: Option<Arc<dyn Fn(String) + Send + Sync>>,
    /// Called with `"listening"` / `"transcribing"` / `"idle"`.
    pub on_status: Option<Arc<dyn Fn(&str) + Send + Sync>>,
    /// Called once after `CONTINUOUS_NO_SPEECH_LIMIT` silent cycles.
    pub on_silent_limit: Option<Arc<dyn Fn() + Send + Sync>>,
}

// ── Continuous (VAD) state ───────────────────────────────────────────────

struct ContinuousState {
    active: bool,
    callbacks: ContinuousCallbacks,
    no_speech_count: i32,
}

/// A `threading.Event`-equivalent: set/clear with a `wait(timeout)` that blocks
/// until set. Used as the TTS-vs-STT feedback guard.
struct EventFlag {
    set: Mutex<bool>,
    cvar: Condvar,
}

impl EventFlag {
    fn new(initial: bool) -> Self {
        EventFlag {
            set: Mutex::new(initial),
            cvar: Condvar::new(),
        }
    }

    fn set(&self) {
        let mut g = self.set.lock().unwrap();
        *g = true;
        self.cvar.notify_all();
    }

    fn clear(&self) {
        let mut g = self.set.lock().unwrap();
        *g = false;
    }

    fn is_set(&self) -> bool {
        *self.set.lock().unwrap()
    }

    /// Block until set or `timeout` elapses. Returns the set state.
    fn wait_timeout(&self, timeout: Duration) -> bool {
        let mut g = self.set.lock().unwrap();
        if *g {
            return true;
        }
        let (guard, _res) = self.cvar.wait_timeout(g, timeout).unwrap();
        g = guard;
        *g
    }
}

/// The process-wide voice manager. Holds all module-level state that
/// `voice.py` kept in globals (recorders, locks, the TTS feedback Event,
/// callbacks) behind injected hardware backends.
///
/// Cloneable cheaply (`Arc`-backed) so the same manager can be shared across
/// the gateway threads.
#[derive(Clone)]
pub struct VoiceManager {
    inner: Arc<VoiceInner>,
}

struct VoiceInner {
    // Push-to-talk state.
    recorder: Mutex<Option<Box<dyn AudioRecorder>>>,

    // Continuous (VAD) state.
    continuous_recorder: Mutex<Option<Box<dyn AudioRecorder>>>,
    continuous: Mutex<ContinuousState>,

    // TTS-vs-STT feedback guard. Cleared while speak_text is playing, set
    // while silent. Initially "not playing".
    tts_playing: EventFlag,

    // Injected backends.
    recorder_factory: Box<dyn Fn() -> Box<dyn AudioRecorder> + Send + Sync>,
    transcriber: Arc<dyn Transcriber>,
    beeper: Arc<dyn Beeper>,
    tts: Arc<dyn TtsBackend>,

    debug: AtomicBool,
}

/// Emit a debug breadcrumb when `HERMES_VOICE_DEBUG=1`.
///
/// Goes to stderr so the TUI gateway wraps it as a `gateway.stderr` event. A
/// broken stderr pipe must not kill the gateway, so write errors are swallowed.
fn debug_enabled() -> bool {
    std::env::var("HERMES_VOICE_DEBUG")
        .map(|v| v.trim() == "1")
        .unwrap_or(false)
}

impl VoiceInner {
    fn debug(&self, msg: &str) {
        if !self.debug.load(Ordering::Relaxed) && !debug_enabled() {
            return;
        }
        // Best-effort: ignore broken pipes.
        let _ = std::io::Write::write_all(
            &mut std::io::stderr(),
            format!("[voice] {msg}\n").as_bytes(),
        );
        let _ = std::io::Write::flush(&mut std::io::stderr());
    }

    /// Audible cue matching cli.py's record/stop beeps. Best-effort.
    fn play_beep(&self, frequency: i32, count: i32) {
        if !self.tts.beeps_enabled() {
            // Beep-enabled state is sourced from the TTS backend's config view;
            // matches `_beeps_enabled()` reading voice.beep_enabled.
        }
        if !self.beeps_enabled() {
            return;
        }
        // play_beep is best-effort; errors are swallowed by the trait impl.
        self.beeper.play_beep(frequency, count);
    }

    fn beeps_enabled(&self) -> bool {
        self.tts.beeps_enabled()
    }
}

impl VoiceManager {
    /// Construct a manager from concrete hardware backends.
    pub fn new(
        recorder_factory: Box<dyn Fn() -> Box<dyn AudioRecorder> + Send + Sync>,
        transcriber: Arc<dyn Transcriber>,
        beeper: Arc<dyn Beeper>,
        tts: Arc<dyn TtsBackend>,
    ) -> Self {
        let tts_playing = EventFlag::new(true); // initially "not playing"
        VoiceManager {
            inner: Arc::new(VoiceInner {
                recorder: Mutex::new(None),
                continuous_recorder: Mutex::new(None),
                continuous: Mutex::new(ContinuousState {
                    active: false,
                    callbacks: ContinuousCallbacks::default(),
                    no_speech_count: 0,
                }),
                tts_playing,
                recorder_factory,
                transcriber,
                beeper,
                tts,
                debug: AtomicBool::new(false),
            }),
        }
    }

    /// Force-enable debug logging regardless of the env var (test/diag aid).
    pub fn set_debug(&self, on: bool) {
        self.inner.debug.store(on, Ordering::Relaxed);
    }

    // ── Push-to-talk API ─────────────────────────────────────────────────

    /// Begin capturing from the default input device (push-to-talk).
    ///
    /// Idempotent — calling again while a recording is in progress is a no-op.
    pub fn start_recording(&self) -> Result<(), String> {
        let mut guard = self.inner.recorder.lock().unwrap();
        if let Some(rec) = guard.as_ref() {
            if rec.is_recording() {
                return Ok(());
            }
        }
        let mut rec = (self.inner.recorder_factory)();
        rec.start(None)?;
        *guard = Some(rec);
        Ok(())
    }

    /// Stop the active push-to-talk recording, transcribe, return text.
    ///
    /// Returns `None` when no recording is active, when the microphone captured
    /// no speech, or when Whisper returned a known hallucination.
    pub fn stop_and_transcribe(&self) -> Option<String> {
        let mut rec = {
            let mut guard = self.inner.recorder.lock().unwrap();
            guard.take()
        }?;

        let wav_path = rec.stop()?;

        let result = match self.inner.transcriber.transcribe(&wav_path) {
            Ok(r) => Some(r),
            Err(e) => {
                log::warn!("voice transcription failed: {e}");
                None
            }
        };
        cleanup_wav(&wav_path);
        let result = result?;

        // transcribe_recording returns {"success": bool, "transcript": str, ...}
        if !result
            .get("success")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            return None;
        }
        let text = result
            .get("transcript")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if text.is_empty() || self.inner.transcriber.is_hallucination(&text) {
            return None;
        }
        Some(text)
    }

    // ── Continuous (VAD) API ─────────────────────────────────────────────

    /// Start a VAD-driven continuous recording loop.
    ///
    /// The loop calls `on_transcript(text)` each time speech is detected and
    /// transcribed successfully, then auto-restarts. After
    /// `CONTINUOUS_NO_SPEECH_LIMIT` consecutive silent cycles the loop stops
    /// itself and calls `on_silent_limit`. Idempotent — calling while already
    /// active is a no-op.
    pub fn start_continuous(
        &self,
        callbacks: ContinuousCallbacks,
        silence_threshold: i64,
        silence_duration: f64,
    ) -> Result<(), String> {
        let on_status: Option<Arc<dyn Fn(&str) + Send + Sync>>;
        {
            let mut cont = self.inner.continuous.lock().unwrap();
            if cont.active {
                self.inner.debug("start_continuous: already active — no-op");
                return Ok(());
            }
            cont.active = true;
            on_status = callbacks.on_status.clone();
            cont.callbacks = callbacks;
            cont.no_speech_count = 0;

            let mut rec_guard = self.inner.continuous_recorder.lock().unwrap();
            if rec_guard.is_none() {
                *rec_guard = Some((self.inner.recorder_factory)());
            }
            let rec = rec_guard.as_mut().unwrap();
            rec.set_silence_threshold(silence_threshold);
            rec.set_silence_duration(silence_duration);
        }

        self.inner.debug(&format!(
            "start_continuous: begin (threshold={silence_threshold}, duration={silence_duration}s)"
        ));

        // CLI parity: single 880 Hz beep *before* opening the stream — placing
        // the beep after stream.start() on macOS triggers a CoreAudio conflict.
        self.inner.play_beep(880, 1);

        let cb = self.silence_callback();
        let start_res = {
            let mut rec_guard = self.inner.continuous_recorder.lock().unwrap();
            rec_guard.as_mut().unwrap().start(Some(cb))
        };
        if let Err(e) = start_res {
            log::error!("failed to start continuous recording: {e}");
            self.inner
                .debug(&format!("start_continuous: rec.start raised {e}"));
            self.inner.continuous.lock().unwrap().active = false;
            return Err(e);
        }

        if let Some(cb) = on_status {
            cb("listening");
        }
        Ok(())
    }

    /// Stop the active continuous loop and release the microphone.
    ///
    /// Idempotent. Any in-flight transcription completes but its result is
    /// discarded (the callback checks `active` before firing).
    pub fn stop_continuous(&self) {
        let on_status: Option<Arc<dyn Fn(&str) + Send + Sync>>;
        {
            let mut cont = self.inner.continuous.lock().unwrap();
            if !cont.active {
                return;
            }
            cont.active = false;
            on_status = cont.callbacks.on_status.clone();
            cont.callbacks = ContinuousCallbacks::default();
            cont.no_speech_count = 0;
        }

        // cancel() (not stop()) discards buffered frames.
        {
            let mut rec_guard = self.inner.continuous_recorder.lock().unwrap();
            if let Some(rec) = rec_guard.as_mut() {
                rec.cancel();
            }
        }

        // Audible "recording stopped" cue.
        self.inner.play_beep(660, 2);

        if let Some(cb) = on_status {
            cb("idle");
        }
    }

    /// Whether a continuous voice loop is currently running.
    pub fn is_continuous_active(&self) -> bool {
        self.inner.continuous.lock().unwrap().active
    }

    /// Build the silence callback closure, capturing a manager handle.
    fn silence_callback(&self) -> SilenceCallback {
        let this = self.clone();
        Arc::new(move || this.on_silence())
    }

    /// AudioRecorder silence callback — runs in a daemon thread.
    ///
    /// Stops the current capture, transcribes, delivers the text, and — if the
    /// loop is still active — starts the next capture. Three consecutive silent
    /// cycles end the loop.
    fn on_silence(&self) {
        self.inner.debug("_continuous_on_silence: fired");

        let (on_transcript, on_status, on_silent_limit) = {
            let cont = self.inner.continuous.lock().unwrap();
            if !cont.active {
                self.inner
                    .debug("_continuous_on_silence: loop inactive — abort");
                return;
            }
            (
                cont.callbacks.on_transcript.clone(),
                cont.callbacks.on_status.clone(),
                cont.callbacks.on_silent_limit.clone(),
            )
        };

        // Confirm a recorder exists.
        {
            let rec_guard = self.inner.continuous_recorder.lock().unwrap();
            if rec_guard.is_none() {
                self.inner.debug("_continuous_on_silence: no recorder — abort");
                return;
            }
        }

        if let Some(cb) = &on_status {
            cb("transcribing");
        }

        let (wav_path, peak_rms) = {
            let mut rec_guard = self.inner.continuous_recorder.lock().unwrap();
            let rec = rec_guard.as_mut().unwrap();
            let wav = rec.stop();
            (wav, rec.peak_rms())
        };
        self.inner.debug(&format!(
            "_continuous_on_silence: rec.stop -> {wav_path:?} (peak_rms={peak_rms})"
        ));

        // CLI parity: double 660 Hz beep after the stream stops.
        self.inner.play_beep(660, 2);

        let mut transcript: Option<String> = None;

        if let Some(ref wav) = wav_path {
            match self.inner.transcriber.transcribe(wav) {
                Ok(result) => {
                    let success = result
                        .get("success")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    let text = result
                        .get("transcript")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    let err = result.get("error");
                    self.inner.debug(&format!(
                        "_continuous_on_silence: transcribe -> success={success} text={text:?} err={err:?}"
                    ));
                    if success && !text.is_empty() && !self.inner.transcriber.is_hallucination(&text)
                    {
                        transcript = Some(text);
                    }
                }
                Err(e) => {
                    log::warn!("continuous transcription failed: {e}");
                    self.inner
                        .debug(&format!("_continuous_on_silence: transcribe raised {e}"));
                }
            }
            cleanup_wav(wav);
        }

        let (should_halt, no_speech) = {
            let mut cont = self.inner.continuous.lock().unwrap();
            if !cont.active {
                self.inner
                    .debug("_continuous_on_silence: stopped during transcribe — no restart");
                return;
            }
            if transcript.is_some() {
                cont.no_speech_count = 0;
            } else {
                cont.no_speech_count += 1;
            }
            (
                cont.no_speech_count >= CONTINUOUS_NO_SPEECH_LIMIT,
                cont.no_speech_count,
            )
        };

        if let (Some(text), Some(cb)) = (&transcript, &on_transcript) {
            // Note: Python catches exceptions here; closures in Rust can't
            // "raise" across the boundary so the call is direct.
            cb(text.clone());
        }

        if should_halt {
            self.inner.debug(&format!(
                "_continuous_on_silence: {no_speech} silent cycles — halting"
            ));
            {
                let mut cont = self.inner.continuous.lock().unwrap();
                cont.active = false;
                cont.no_speech_count = 0;
            }
            if let Some(cb) = &on_silent_limit {
                cb();
            }
            {
                let mut rec_guard = self.inner.continuous_recorder.lock().unwrap();
                if let Some(rec) = rec_guard.as_mut() {
                    rec.cancel();
                }
            }
            if let Some(cb) = &on_status {
                cb("idle");
            }
            return;
        }

        // CLI parity: wait for any in-flight TTS to finish before re-arming the
        // mic, then leave a small gap to avoid catching the speaker tail.
        if !self.inner.tts_playing.is_set() {
            self.inner
                .debug("_continuous_on_silence: waiting for TTS to finish");
            self.inner.tts_playing.wait_timeout(Duration::from_secs(60));
            std::thread::sleep(Duration::from_millis(300));

            // User may have stopped the loop during the wait.
            let cont = self.inner.continuous.lock().unwrap();
            if !cont.active {
                self.inner
                    .debug("_continuous_on_silence: stopped while waiting for TTS");
                return;
            }
        }

        // Restart for the next turn.
        self.inner.debug(&format!(
            "_continuous_on_silence: restarting loop (no_speech={no_speech})"
        ));
        self.inner.play_beep(880, 1);
        let cb = self.silence_callback();
        let start_res = {
            let mut rec_guard = self.inner.continuous_recorder.lock().unwrap();
            rec_guard.as_mut().unwrap().start(Some(cb))
        };
        if let Err(e) = start_res {
            log::error!("failed to restart continuous recording: {e}");
            self.inner
                .debug(&format!("_continuous_on_silence: restart raised {e}"));
            self.inner.continuous.lock().unwrap().active = false;
            return;
        }

        if let Some(cb) = &on_status {
            cb("listening");
        }
    }

    // ── TTS API ───────────────────────────────────────────────────────────

    /// Synthesize `text` with the configured TTS provider and play it.
    ///
    /// Mirrors cli.py:_voice_speak_response exactly — same markdown strip
    /// pipeline, same 4000-char cap, same explicit mp3 output path, same
    /// MP3-over-OGG playback choice, same cleanup of both extensions.
    ///
    /// While playback is in flight the `tts_playing` flag is cleared so the
    /// continuous loop knows to wait before re-arming the mic.
    pub fn speak_text(&self, text: &str) {
        if text.trim().is_empty() {
            return;
        }

        // Cancel any live capture before we open the speakers.
        let mut paused_recording = false;
        {
            let cont = self.inner.continuous.lock().unwrap();
            let active = cont.active;
            drop(cont);
            if active {
                let mut rec_guard = self.inner.continuous_recorder.lock().unwrap();
                if let Some(rec) = rec_guard.as_mut() {
                    if rec.is_recording() {
                        rec.cancel();
                        paused_recording = true;
                    }
                }
            }
        }

        self.inner.tts_playing.clear();
        self.inner.debug(&format!(
            "speak_text: TTS begin (paused_recording={paused_recording})"
        ));

        // Body of the try/finally — run to completion, then always restore the
        // flag + re-arm in the finally section below.
        self.speak_text_body(text);

        // finally:
        self.inner.tts_playing.set();
        self.inner.debug("speak_text: TTS done");

        // Re-arm the mic so the user can answer without pressing Ctrl+B.
        if paused_recording {
            std::thread::sleep(Duration::from_millis(300));
            let cb = self.silence_callback();
            let mut rec_guard = self.inner.continuous_recorder.lock().unwrap();
            let cont = self.inner.continuous.lock().unwrap();
            if cont.active {
                if let Some(rec) = rec_guard.as_mut() {
                    match rec.start(Some(cb)) {
                        Ok(()) => self.inner.debug("speak_text: recording resumed after TTS"),
                        Err(e) => log::warn!("failed to resume recorder after TTS: {e}"),
                    }
                }
            }
        }
    }

    /// The body of [`speak_text`], factored so the caller can wrap it in the
    /// equivalent of Python's `try/finally`.
    fn speak_text_body(&self, text: &str) {
        let tts_text = strip_markdown_for_tts(text);
        if tts_text.is_empty() {
            return;
        }

        // MP3 output path, pre-chosen so we can play the MP3 directly even when
        // text_to_speech auto-converts to OGG for messaging platforms.
        let dir = std::env::temp_dir().join("hermes_voice");
        if let Err(e) = std::fs::create_dir_all(&dir) {
            log::warn!("Voice TTS playback failed: {e}");
            self.inner.debug(&format!("speak_text raised {e}"));
            return;
        }
        let stamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
        let mp3_path = dir.join(format!("tts_{stamp}.mp3"));
        let mp3_str = mp3_path.to_string_lossy().to_string();

        self.inner.debug(&format!(
            "speak_text: synthesizing {} chars -> {mp3_str}",
            tts_text.chars().count()
        ));

        if let Err(e) = self.inner.tts.text_to_speech(&tts_text, &mp3_str) {
            log::warn!("Voice TTS playback failed: {e}");
            self.inner.debug(&format!("speak_text raised {e}"));
            return;
        }

        let size = std::fs::metadata(&mp3_path).map(|m| m.len()).unwrap_or(0);
        if mp3_path.is_file() && size > 0 {
            self.inner
                .debug(&format!("speak_text: playing {mp3_str} ({size} bytes)"));
            self.inner.tts.play_audio_file(&mp3_str);
            let _ = std::fs::remove_file(&mp3_path);
            // ogg sibling: mp3_path.rsplit('.',1)[0] + ".ogg"
            let ogg_path = swap_extension_to_ogg(&mp3_str);
            if std::path::Path::new(&ogg_path).is_file() {
                let _ = std::fs::remove_file(&ogg_path);
            }
        } else {
            self.inner
                .debug(&format!("speak_text: TTS tool produced no audio at {mp3_str}"));
        }
    }
}

/// Best-effort WAV cleanup matching the Python `os.path.isfile` + `os.unlink`
/// guard.
fn cleanup_wav(wav_path: &str) {
    let p = std::path::Path::new(wav_path);
    if p.is_file() {
        let _ = std::fs::remove_file(p);
    }
}

/// Replace the final `.ext` of a path with `.ogg`, matching
/// `mp3_path.rsplit(".", 1)[0] + ".ogg"`.
fn swap_extension_to_ogg(path: &str) -> String {
    match path.rfind('.') {
        Some(idx) => format!("{}.ogg", &path[..idx]),
        None => format!("{path}.ogg"),
    }
}

/// The exact markdown-strip pipeline from cli.py:_voice_speak_response.
///
/// 1. Cap to 4000 chars. 2. Strip fenced code, links, bare URLs, bold, italic,
/// inline code, headers, list bullets, horizontal rules, excess newlines.
/// 3. Trim.
pub fn strip_markdown_for_tts(text: &str) -> String {
    use regex::Regex;

    // Char-based 4000 cap (Python slices by code points).
    let mut tts_text: String = if text.chars().count() > 4000 {
        text.chars().take(4000).collect()
    } else {
        text.to_string()
    };

    // fenced code blocks: ```...``` (DOTALL)
    let re_fence = Regex::new(r"(?s)```.*?```").unwrap();
    tts_text = re_fence.replace_all(&tts_text, " ").into_owned();

    // [text](url) → text
    let re_link = Regex::new(r"\[([^\]]+)\]\([^)]+\)").unwrap();
    tts_text = re_link.replace_all(&tts_text, "$1").into_owned();

    // bare URLs
    let re_url = Regex::new(r"https?://\S+").unwrap();
    tts_text = re_url.replace_all(&tts_text, "").into_owned();

    // bold **...**
    let re_bold = Regex::new(r"(?s)\*\*(.+?)\*\*").unwrap();
    tts_text = re_bold.replace_all(&tts_text, "$1").into_owned();

    // italic *...*
    let re_italic = Regex::new(r"(?s)\*(.+?)\*").unwrap();
    tts_text = re_italic.replace_all(&tts_text, "$1").into_owned();

    // inline code `...`
    let re_code = Regex::new(r"(?s)`(.+?)`").unwrap();
    tts_text = re_code.replace_all(&tts_text, "$1").into_owned();

    // headers: ^#+\s*  (MULTILINE)
    let re_header = Regex::new(r"(?m)^#+\s*").unwrap();
    tts_text = re_header.replace_all(&tts_text, "").into_owned();

    // list bullets: ^\s*[-*]\s+  (MULTILINE)
    let re_bullet = Regex::new(r"(?m)^\s*[-*]\s+").unwrap();
    tts_text = re_bullet.replace_all(&tts_text, "").into_owned();

    // horizontal rules: ---+
    let re_hr = Regex::new(r"---+").unwrap();
    tts_text = re_hr.replace_all(&tts_text, "").into_owned();

    // excess newlines: \n{3,} → \n\n
    let re_nl = Regex::new(r"\n{3,}").unwrap();
    tts_text = re_nl.replace_all(&tts_text, "\n\n").into_owned();

    tts_text.trim().to_string()
}

/// CLI parity helper used by `_beeps_enabled`: read `voice.beep_enabled` from a
/// config map (default `true`).
pub fn beeps_enabled_from_config(cfg: &Value) -> bool {
    cfg.as_object()
        .and_then(|o| o.get("voice"))
        .and_then(|v| v.as_object())
        .and_then(|v| v.get("beep_enabled"))
        .and_then(|b| b.as_bool())
        .unwrap_or(true)
}

/// Reserved-modifier helper exposed for the CLI binding site to warn about
/// TUI-only spellings (`super`/`win`/`windows`).
pub fn is_tui_only_modifier(token: &str) -> bool {
    matches!(token.trim().to_lowercase().as_str(), "super" | "win" | "windows")
}

/// Set of reserved ctrl chars, exposed for callers that mirror the TUI parser.
pub fn reserved_ctrl_chars() -> HashSet<&'static str> {
    ["c", "d", "l"].into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicI32, Ordering};

    // ── Config-parsing helpers ────────────────────────────────────────────

    #[test]
    fn record_key_from_config_shapes() {
        assert_eq!(
            voice_record_key_from_config(&json!({"voice": {"record_key": "ctrl+o"}})),
            Some(json!("ctrl+o"))
        );
        // voice is a bool → None
        assert_eq!(voice_record_key_from_config(&json!({"voice": true})), None);
        // voice is a str → None
        assert_eq!(
            voice_record_key_from_config(&json!({"voice": "cmd+b"})),
            None
        );
        // missing voice
        assert_eq!(voice_record_key_from_config(&json!({})), None);
        // non-object cfg
        assert_eq!(voice_record_key_from_config(&json!("not a dict")), None);
        // voice dict but no record_key
        assert_eq!(
            voice_record_key_from_config(&json!({"voice": {}})),
            None
        );
    }

    #[test]
    fn normalize_defaults_and_single_char() {
        assert_eq!(normalize_voice_record_key_for_prompt_toolkit(&json!(42)), "c-b");
        assert_eq!(normalize_voice_record_key_for_prompt_toolkit(&json!("")), "c-b");
        assert_eq!(normalize_voice_record_key_for_prompt_toolkit(&json!("   ")), "c-b");
        // bare char
        assert_eq!(normalize_voice_record_key_for_prompt_toolkit(&json!("b")), "c-b");
        // multi-modifier
        assert_eq!(
            normalize_voice_record_key_for_prompt_toolkit(&json!("ctrl+alt+r")),
            "c-b"
        );
        // single char ok
        assert_eq!(
            normalize_voice_record_key_for_prompt_toolkit(&json!("ctrl+o")),
            "c-o"
        );
        assert_eq!(
            normalize_voice_record_key_for_prompt_toolkit(&json!("CTRL+O")),
            "c-o"
        );
        assert_eq!(
            normalize_voice_record_key_for_prompt_toolkit(&json!("control+r")),
            "c-r"
        );
        assert_eq!(
            normalize_voice_record_key_for_prompt_toolkit(&json!("alt+r")),
            "a-r"
        );
        assert_eq!(
            normalize_voice_record_key_for_prompt_toolkit(&json!("option+r")),
            "a-r"
        );
    }

    #[test]
    fn normalize_reserved_and_named() {
        // reserved ctrl
        for c in ["c", "d", "l"] {
            assert_eq!(
                normalize_voice_record_key_for_prompt_toolkit(&json!(format!("ctrl+{c}"))),
                "c-b"
            );
        }
        // named keys + alias collapse
        assert_eq!(
            normalize_voice_record_key_for_prompt_toolkit(&json!("ctrl+space")),
            "c-space"
        );
        assert_eq!(
            normalize_voice_record_key_for_prompt_toolkit(&json!("ctrl+return")),
            "c-enter"
        );
        assert_eq!(
            normalize_voice_record_key_for_prompt_toolkit(&json!("alt+spc")),
            "a-space"
        );
        // typo'd named key
        assert_eq!(
            normalize_voice_record_key_for_prompt_toolkit(&json!("ctrl+spcae")),
            "c-b"
        );
        // unknown modifier
        assert_eq!(
            normalize_voice_record_key_for_prompt_toolkit(&json!("hyper+r")),
            "c-b"
        );
        // super/win/windows → default
        for m in ["super", "win", "windows"] {
            assert_eq!(
                normalize_voice_record_key_for_prompt_toolkit(&json!(format!("{m}+b"))),
                "c-b"
            );
        }
    }

    #[test]
    fn format_for_status_cases() {
        assert_eq!(format_voice_record_key_for_status(&json!("ctrl+o")), "Ctrl+O");
        assert_eq!(format_voice_record_key_for_status(&json!("alt+space")), "Alt+Space");
        assert_eq!(format_voice_record_key_for_status(&json!("ctrl+return")), "Ctrl+Enter");
        assert_eq!(format_voice_record_key_for_status(&json!(42)), "Ctrl+B");
        assert_eq!(format_voice_record_key_for_status(&json!("ctrl+c")), "Ctrl+B");
    }

    #[test]
    fn beeps_enabled_defaults() {
        assert!(beeps_enabled_from_config(&json!({})));
        assert!(beeps_enabled_from_config(&json!({"voice": {}})));
        assert!(beeps_enabled_from_config(&json!({"voice": {"beep_enabled": true}})));
        assert!(!beeps_enabled_from_config(&json!({"voice": {"beep_enabled": false}})));
        // malformed voice → default true
        assert!(beeps_enabled_from_config(&json!({"voice": "x"})));
    }

    #[test]
    fn markdown_strip_pipeline() {
        let out = strip_markdown_for_tts("# Title\n\n**bold** and *italic* and `code`");
        assert_eq!(out, "Title\n\nbold and italic and code");

        let out2 = strip_markdown_for_tts("see [docs](http://x.com) here");
        assert_eq!(out2, "see docs here");

        let out3 = strip_markdown_for_tts("visit https://example.com/page now");
        assert_eq!(out3, "visit  now");

        let out4 = strip_markdown_for_tts("```\ncode\n```\nafter");
        assert_eq!(out4, "after");

        let out5 = strip_markdown_for_tts("- one\n- two");
        assert_eq!(out5, "one\ntwo");

        assert_eq!(strip_markdown_for_tts("   "), "");
    }

    #[test]
    fn markdown_4000_cap() {
        let big = "a".repeat(5000);
        let out = strip_markdown_for_tts(&big);
        assert_eq!(out.chars().count(), 4000);
    }

    #[test]
    fn swap_ext() {
        assert_eq!(swap_extension_to_ogg("/tmp/x/tts_1.mp3"), "/tmp/x/tts_1.ogg");
        assert_eq!(swap_extension_to_ogg("/tmp/noext"), "/tmp/noext.ogg");
    }

    // ── Stateful loop tests with fake backends ────────────────────────────

    struct FakeRecorder {
        recording: Arc<AtomicBool>,
        // queue of stop() return values consumed FIFO
        stop_queue: Arc<Mutex<Vec<Option<String>>>>,
        start_count: Arc<AtomicI32>,
        // when started with a silence callback, store it so a test can fire it
        last_cb: Arc<Mutex<Option<SilenceCallback>>>,
    }

    impl AudioRecorder for FakeRecorder {
        fn is_recording(&self) -> bool {
            self.recording.load(Ordering::SeqCst)
        }
        fn set_silence_threshold(&mut self, _t: i64) {}
        fn set_silence_duration(&mut self, _d: f64) {}
        fn start(&mut self, on_silence_stop: Option<SilenceCallback>) -> Result<(), String> {
            self.recording.store(true, Ordering::SeqCst);
            self.start_count.fetch_add(1, Ordering::SeqCst);
            *self.last_cb.lock().unwrap() = on_silence_stop;
            Ok(())
        }
        fn stop(&mut self) -> Option<String> {
            self.recording.store(false, Ordering::SeqCst);
            let mut q = self.stop_queue.lock().unwrap();
            if q.is_empty() {
                None
            } else {
                q.remove(0)
            }
        }
        fn cancel(&mut self) {
            self.recording.store(false, Ordering::SeqCst);
        }
    }

    struct FakeTranscriber {
        // map wav path -> transcript value
        responses: Mutex<Vec<Value>>,
    }
    impl Transcriber for FakeTranscriber {
        fn transcribe(&self, _wav: &str) -> Result<Value, String> {
            let mut r = self.responses.lock().unwrap();
            if r.is_empty() {
                Ok(json!({"success": false, "transcript": ""}))
            } else {
                Ok(r.remove(0))
            }
        }
        fn is_hallucination(&self, text: &str) -> bool {
            text == "Thank you."
        }
    }

    struct NoBeep;
    impl Beeper for NoBeep {
        fn play_beep(&self, _f: i32, _c: i32) {}
    }

    struct NoTts;
    impl TtsBackend for NoTts {
        fn text_to_speech(&self, _t: &str, _p: &str) -> Result<(), String> {
            Ok(())
        }
        fn play_audio_file(&self, _p: &str) {}
        fn beeps_enabled(&self) -> bool {
            false
        }
    }

    fn build_manager(
        recording: Arc<AtomicBool>,
        stop_queue: Arc<Mutex<Vec<Option<String>>>>,
        start_count: Arc<AtomicI32>,
        last_cb: Arc<Mutex<Option<SilenceCallback>>>,
        transcripts: Vec<Value>,
    ) -> VoiceManager {
        let rec_recording = recording.clone();
        let rec_stop = stop_queue.clone();
        let rec_count = start_count.clone();
        let rec_cb = last_cb.clone();
        let factory = Box::new(move || {
            Box::new(FakeRecorder {
                recording: rec_recording.clone(),
                stop_queue: rec_stop.clone(),
                start_count: rec_count.clone(),
                last_cb: rec_cb.clone(),
            }) as Box<dyn AudioRecorder>
        });
        VoiceManager::new(
            factory,
            Arc::new(FakeTranscriber {
                responses: Mutex::new(transcripts),
            }),
            Arc::new(NoBeep),
            Arc::new(NoTts),
        )
    }

    #[test]
    fn push_to_talk_idempotent_and_transcribe() {
        let recording = Arc::new(AtomicBool::new(false));
        let stop_queue = Arc::new(Mutex::new(vec![Some("/tmp/no_such_voice_test.wav".to_string())]));
        let start_count = Arc::new(AtomicI32::new(0));
        let last_cb = Arc::new(Mutex::new(None));
        let mgr = build_manager(
            recording.clone(),
            stop_queue,
            start_count.clone(),
            last_cb,
            vec![json!({"success": true, "transcript": "hello world"})],
        );

        mgr.start_recording().unwrap();
        // idempotent: second call while recording is a no-op (no new start)
        mgr.start_recording().unwrap();
        assert_eq!(start_count.load(Ordering::SeqCst), 1);

        let text = mgr.stop_and_transcribe();
        assert_eq!(text, Some("hello world".to_string()));

        // No active recording now → None.
        assert_eq!(mgr.stop_and_transcribe(), None);
    }

    #[test]
    fn push_to_talk_hallucination_filtered() {
        let recording = Arc::new(AtomicBool::new(false));
        let stop_queue = Arc::new(Mutex::new(vec![Some("/tmp/no_such_voice_test2.wav".to_string())]));
        let mgr = build_manager(
            recording,
            stop_queue,
            Arc::new(AtomicI32::new(0)),
            Arc::new(Mutex::new(None)),
            vec![json!({"success": true, "transcript": "Thank you."})],
        );
        mgr.start_recording().unwrap();
        assert_eq!(mgr.stop_and_transcribe(), None);
    }

    #[test]
    fn continuous_start_stop_idempotent() {
        let recording = Arc::new(AtomicBool::new(false));
        let mgr = build_manager(
            recording,
            Arc::new(Mutex::new(vec![])),
            Arc::new(AtomicI32::new(0)),
            Arc::new(Mutex::new(None)),
            vec![],
        );
        assert!(!mgr.is_continuous_active());
        mgr.start_continuous(ContinuousCallbacks::default(), 200, 3.0)
            .unwrap();
        assert!(mgr.is_continuous_active());
        // idempotent
        mgr.start_continuous(ContinuousCallbacks::default(), 200, 3.0)
            .unwrap();
        assert!(mgr.is_continuous_active());

        mgr.stop_continuous();
        assert!(!mgr.is_continuous_active());
        // idempotent stop
        mgr.stop_continuous();
        assert!(!mgr.is_continuous_active());
    }

    #[test]
    fn continuous_silent_limit_halts() {
        use std::sync::atomic::AtomicUsize;
        let recording = Arc::new(AtomicBool::new(false));
        let last_cb = Arc::new(Mutex::new(None));
        // 3 silent transcribes → halt
        let mgr = build_manager(
            recording,
            Arc::new(Mutex::new(vec![Some("/tmp/x1.wav".into()), Some("/tmp/x2.wav".into()), Some("/tmp/x3.wav".into())])),
            Arc::new(AtomicI32::new(0)),
            last_cb.clone(),
            vec![
                json!({"success": false, "transcript": ""}),
                json!({"success": false, "transcript": ""}),
                json!({"success": false, "transcript": ""}),
            ],
        );

        let silent_fired = Arc::new(AtomicUsize::new(0));
        let sf = silent_fired.clone();
        let callbacks = ContinuousCallbacks {
            on_transcript: None,
            on_status: None,
            on_silent_limit: Some(Arc::new(move || {
                sf.fetch_add(1, Ordering::SeqCst);
            })),
        };
        mgr.start_continuous(callbacks, 200, 3.0).unwrap();

        // Fire silence 3 times by invoking the stored callback.
        for _ in 0..3 {
            let cb = last_cb.lock().unwrap().clone();
            if let Some(cb) = cb {
                cb();
            }
        }
        assert_eq!(silent_fired.load(Ordering::SeqCst), 1);
        assert!(!mgr.is_continuous_active());
    }

    #[test]
    fn continuous_transcript_resets_count_and_restarts() {
        let recording = Arc::new(AtomicBool::new(false));
        let last_cb = Arc::new(Mutex::new(None));
        let start_count = Arc::new(AtomicI32::new(0));
        let got = Arc::new(Mutex::new(Vec::<String>::new()));
        let g2 = got.clone();
        let mgr = build_manager(
            recording,
            Arc::new(Mutex::new(vec![Some("/tmp/y1.wav".into())])),
            start_count.clone(),
            last_cb.clone(),
            vec![json!({"success": true, "transcript": "hi there"})],
        );
        let callbacks = ContinuousCallbacks {
            on_transcript: Some(Arc::new(move |t| g2.lock().unwrap().push(t))),
            on_status: None,
            on_silent_limit: None,
        };
        mgr.start_continuous(callbacks, 200, 3.0).unwrap();
        let initial_starts = start_count.load(Ordering::SeqCst);

        let cb = last_cb.lock().unwrap().clone().unwrap();
        cb();

        assert_eq!(*got.lock().unwrap(), vec!["hi there".to_string()]);
        assert!(mgr.is_continuous_active());
        // restarted for next turn
        assert!(start_count.load(Ordering::SeqCst) > initial_starts);
    }

    #[test]
    fn event_flag_wait_returns_immediately_when_set() {
        let f = EventFlag::new(true);
        assert!(f.wait_timeout(Duration::from_secs(5)));
        f.clear();
        assert!(!f.is_set());
        f.set();
        assert!(f.is_set());
    }

    #[test]
    fn is_tui_only_modifier_check() {
        assert!(is_tui_only_modifier("super"));
        assert!(is_tui_only_modifier(" WIN "));
        assert!(!is_tui_only_modifier("ctrl"));
    }
}
