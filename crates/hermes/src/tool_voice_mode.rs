//! Voice Mode -- push-to-talk audio recording and playback.
//!
//! Native Rust port of `tools/voice_mode.py`.
//!
//! The original Python module relies on `sounddevice`/`numpy`/PortAudio for
//! live microphone capture and on the stdlib `wave` module for WAV encoding.
//! Those C-extension audio backends are not available in this Rust crate, so
//! the live-capture `AudioRecorder` here implements the *pure* logic that the
//! Python class drives -- the silence-detection state machine, RMS tracking,
//! WAV encoding, and recording lifecycle -- against caller-supplied PCM
//! samples instead of a PortAudio callback.  All the non-audio-hardware logic
//! (environment detection, Whisper-hallucination filtering, Termux recorder
//! subprocess wiring, system audio playback, temp-file cleanup) is ported
//! faithfully.
//!
//! Behaviour preserved from Python:
//!   * Recording parameters (16 kHz, mono, int16).
//!   * Silence-detection thresholds and the dip-tolerance / resume state
//!     machine in the audio callback.
//!   * `detect_audio_environment` warnings/notices ordering and wording.
//!   * `is_whisper_hallucination` exact + repeat-regex matching.
//!   * Termux recorder command construction and stop/cancel guards.
//!   * `play_audio_file` system-player fallback ordering per platform.
//!   * `cleanup_temp_recordings` age filter on `recording_*.wav`.

use std::collections::HashSet;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use regex::Regex;

// ---------------------------------------------------------------------------
// Recording parameters
// ---------------------------------------------------------------------------

/// Whisper native sample rate.
pub const SAMPLE_RATE: u32 = 16000;
/// Mono.
pub const CHANNELS: u16 = 1;
/// Bytes per sample (int16).
pub const SAMPLE_WIDTH: u16 = 2;

/// RMS below this == silence (int16 range 0-32767).
pub const SILENCE_RMS_THRESHOLD: i32 = 200;
/// Seconds of continuous silence before auto-stop.
pub const SILENCE_DURATION_SECONDS: f64 = 3.0;

/// Temp directory for voice recordings: `<tmp>/hermes_voice`.
pub fn temp_dir() -> PathBuf {
    std::env::temp_dir().join("hermes_voice")
}

// ---------------------------------------------------------------------------
// Environment helpers (mirror hermes_constants)
// ---------------------------------------------------------------------------

/// Mirror of `hermes_constants.is_termux`.
fn is_termux_environment() -> bool {
    // PREFIX containing com.termux is the canonical signal.
    if let Ok(prefix) = std::env::var("PREFIX") {
        if prefix.contains("com.termux") {
            return true;
        }
    }
    Path::new("/data/data/com.termux/files/usr").exists()
}

/// Mirror of `hermes_constants.is_container` (Docker/Podman detection).
fn is_container() -> bool {
    if Path::new("/.dockerenv").exists() {
        return true;
    }
    if Path::new("/run/.containerenv").exists() {
        return true;
    }
    if std::env::var("container").is_ok() {
        return true;
    }
    if let Ok(cgroup) = fs::read_to_string("/proc/1/cgroup") {
        let lc = cgroup.to_lowercase();
        if lc.contains("docker") || lc.contains("containerd") || lc.contains("podman") {
            return true;
        }
    }
    false
}

/// `shutil.which` equivalent: search PATH for an executable.
fn which(name: &str) -> Option<PathBuf> {
    if name.contains('/') {
        let p = PathBuf::from(name);
        return if p.is_file() { Some(p) } else { None };
    }
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Install hint for the audio-capture libraries.
pub fn voice_capture_install_hint() -> String {
    if is_termux_environment() {
        "pkg install python-numpy portaudio && python -m pip install sounddevice".to_string()
    } else {
        "pip install sounddevice numpy".to_string()
    }
}

/// Path to `termux-microphone-record`, if available in this Termux env.
fn termux_microphone_command() -> Option<PathBuf> {
    if !is_termux_environment() {
        return None;
    }
    which("termux-microphone-record")
}

/// Whether the Termux:API Android app (`com.termux.api`) is installed.
fn termux_api_app_installed() -> bool {
    if !is_termux_environment() {
        return false;
    }
    match Command::new("pm")
        .args(["list", "packages", "com.termux.api"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
    {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            stdout.contains("package:com.termux.api")
        }
        Err(_) => false,
    }
}

/// Whether Termux:API microphone capture is fully available.
pub fn termux_voice_capture_available() -> bool {
    termux_microphone_command().is_some() && termux_api_app_installed()
}

/// Whether the (native) audio stack is available.
///
/// In Python this attempts to import sounddevice/numpy.  This Rust port has no
/// PortAudio backend, so it always reports `false` -- callers should rely on
/// the Termux backend or treat voice capture as unavailable.
pub fn audio_available() -> bool {
    false
}

// ---------------------------------------------------------------------------
// Environment detection
// ---------------------------------------------------------------------------

/// Result of [`detect_audio_environment`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub struct AudioEnvironment {
    /// `true` when no hard-fail warnings are present.
    pub available: bool,
    /// Hard-fail reasons that block voice mode.
    pub warnings: Vec<String>,
    /// Informational messages that do NOT block voice mode.
    pub notices: Vec<String>,
}

/// Detect if the current environment supports audio I/O.
///
/// Returns warnings (hard-fail) and notices (informational).  Mirrors the
/// Python control flow, including the SSH / container / WSL checks and the
/// audio-library availability branch.
pub fn detect_audio_environment() -> AudioEnvironment {
    let mut warnings: Vec<String> = Vec::new();
    let mut notices: Vec<String> = Vec::new();

    let termux_mic_cmd = termux_microphone_command();
    let termux_app_installed = termux_api_app_installed();
    let termux_capture = termux_mic_cmd.is_some() && termux_app_installed;

    // SSH detection
    if ["SSH_CLIENT", "SSH_TTY", "SSH_CONNECTION"]
        .iter()
        .any(|v| std::env::var(v).map(|s| !s.is_empty()).unwrap_or(false))
    {
        warnings.push("Running over SSH -- no audio devices available".to_string());
    }

    // Docker/Podman container detection
    if is_container() {
        warnings.push("Running inside Docker container -- no audio devices".to_string());
    }

    // WSL detection -- PulseAudio bridge makes audio work in WSL.
    if let Ok(version) = fs::read_to_string("/proc/version") {
        if version.to_lowercase().contains("microsoft") {
            if std::env::var("PULSE_SERVER").map(|s| !s.is_empty()).unwrap_or(false) {
                notices.push("Running in WSL with PulseAudio bridge".to_string());
            } else {
                warnings.push(
                    "Running in WSL -- audio requires PulseAudio bridge.\n\
                     \x20 1. Set PULSE_SERVER=unix:/mnt/wslg/PulseServer\n\
                     \x20 2. Create ~/.asoundrc pointing ALSA at PulseAudio\n\
                     \x20 3. Verify with: arecord -d 3 /tmp/test.wav && aplay /tmp/test.wav"
                        .to_string(),
                );
            }
        }
    }

    // Audio library availability.  This port has no PortAudio backend, so we
    // model the Python `ImportError` branch (sounddevice not importable).
    if audio_available() {
        // Unreachable today, but kept for parity: device query succeeded.
    } else if termux_capture {
        notices.push(
            "Termux:API microphone recording available (sounddevice not required)".to_string(),
        );
    } else if termux_mic_cmd.is_some() && !termux_app_installed {
        warnings.push(
            "Termux:API Android app is not installed. Install/update the Termux:API app to use \
             termux-microphone-record."
                .to_string(),
        );
    } else if is_termux_environment() {
        warnings.push(
            "PortAudio system library not found -- install it first:\n\
             \x20 Termux: pkg install portaudio\n\
             Then retry /voice on."
                .to_string(),
        );
    } else {
        warnings.push(format!(
            "Audio libraries not installed ({})",
            voice_capture_install_hint()
        ));
    }

    let available = warnings.is_empty();
    AudioEnvironment {
        available,
        warnings,
        notices,
    }
}

// ---------------------------------------------------------------------------
// Whisper hallucination filter
// ---------------------------------------------------------------------------

/// Phrases Whisper commonly hallucinates on silent/near-silent audio.
pub fn whisper_hallucinations() -> &'static HashSet<&'static str> {
    use std::sync::OnceLock;
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| {
        [
            "thank you.",
            "thank you",
            "thanks for watching.",
            "thanks for watching",
            "subscribe to my channel.",
            "subscribe to my channel",
            "like and subscribe.",
            "like and subscribe",
            "please subscribe.",
            "please subscribe",
            "thank you for watching.",
            "thank you for watching",
            "bye.",
            "bye",
            "you",
            "the end.",
            "the end",
            // Non-English hallucinations (common on silence)
            "продолжение следует",
            "продолжение следует...",
            "sous-titres",
            "sous-titres réalisés par la communauté d'amara.org",
            "sottotitoli creati dalla comunità amara.org",
            "untertitel von stephanie geiges",
            "amara.org",
            "www.mooji.org",
            "ご視聴ありがとうございました",
        ]
        .into_iter()
        .collect()
    })
}

fn hallucination_repeat_re() -> &'static Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)^(?:thank you|thanks|bye|you|ok|okay|the end|\.|\s|,|!)+$").unwrap()
    })
}

/// Check if a transcript is a known Whisper hallucination on silence.
pub fn is_whisper_hallucination(transcript: &str) -> bool {
    let cleaned = transcript.trim().to_lowercase();
    if cleaned.is_empty() {
        return true;
    }
    // Exact match against known phrases (and with trailing `.`/`!` stripped).
    let stripped = cleaned.trim_end_matches(['.', '!']);
    let set = whisper_hallucinations();
    if set.contains(stripped) || set.contains(cleaned.as_str()) {
        return true;
    }
    // Repetitive patterns (e.g. "Thank you. Thank you. Thank you. you").
    if hallucination_repeat_re().is_match(&cleaned) {
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// STT dispatch result type
// ---------------------------------------------------------------------------

/// Result of a transcription, mirroring the Python dict shape.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub struct TranscriptionResult {
    pub success: bool,
    #[serde(default)]
    pub transcript: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub filtered: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// Apply the Whisper-hallucination filter to a raw transcription result.
///
/// Mirrors `transcribe_recording`'s post-processing: when transcription
/// succeeded but the transcript is a known hallucination, blank it out and
/// flag `filtered = true`.  (The actual STT call lives in the transcription
/// tools module; this is the pure post-filter.)
pub fn apply_hallucination_filter(result: TranscriptionResult) -> TranscriptionResult {
    if result.success && is_whisper_hallucination(&result.transcript) {
        log::info!("Filtered Whisper hallucination: {:?}", result.transcript);
        return TranscriptionResult {
            success: true,
            transcript: String::new(),
            error: None,
            filtered: true,
        };
    }
    result
}

// ---------------------------------------------------------------------------
// WAV encoding
// ---------------------------------------------------------------------------

/// Write int16 mono PCM samples to a canonical 16 kHz WAV file inside the temp
/// voice directory, returning the path.  Mirrors `AudioRecorder._write_wav`.
pub fn write_wav(samples: &[i16]) -> std::io::Result<PathBuf> {
    let dir = temp_dir();
    fs::create_dir_all(&dir)?;
    let timestamp = timestamp_now();
    let wav_path = dir.join(format!("recording_{timestamp}.wav"));
    write_wav_to(&wav_path, samples, SAMPLE_RATE, CHANNELS)?;
    let size = fs::metadata(&wav_path).map(|m| m.len()).unwrap_or(0);
    log::info!("WAV written: {} ({} bytes)", wav_path.display(), size);
    Ok(wav_path)
}

/// Write int16 PCM samples to a specific path as a PCM WAV file.
pub fn write_wav_to(
    path: &Path,
    samples: &[i16],
    sample_rate: u32,
    channels: u16,
) -> std::io::Result<()> {
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * channels as u32 * (bits_per_sample as u32 / 8);
    let block_align = channels * (bits_per_sample / 8);
    let data_len = (samples.len() * 2) as u32;

    let mut buf: Vec<u8> = Vec::with_capacity(44 + samples.len() * 2);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_len).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        buf.extend_from_slice(&s.to_le_bytes());
    }

    let mut f = fs::File::create(path)?;
    f.write_all(&buf)?;
    f.flush()?;
    Ok(())
}

/// Compute int16-domain RMS of a PCM chunk (matches numpy `sqrt(mean(x**2))`).
pub fn rms_i16(samples: &[i16]) -> i32 {
    if samples.is_empty() {
        return 0;
    }
    let mut sum = 0.0f64;
    for &s in samples {
        let v = s as f64;
        sum += v * v;
    }
    (sum / samples.len() as f64).sqrt() as i32
}

/// String timestamp in the Python `%Y%m%d_%H%M%S` form.
fn timestamp_now() -> String {
    use chrono::Local;
    Local::now().format("%Y%m%d_%H%M%S").to_string()
}

// ---------------------------------------------------------------------------
// AudioRecorder -- silence-detection state machine
// ---------------------------------------------------------------------------

/// Whether a recorder backend supports silence auto-stop.
pub trait Recorder {
    fn is_recording(&self) -> bool;
    fn elapsed_seconds(&self) -> f64;
    fn current_rms(&self) -> i32;
    fn supports_silence_autostop(&self) -> bool;
}

/// Thread-safe audio recorder driving the silence-detection state machine.
///
/// Unlike the Python class -- which is fed by a PortAudio callback -- this
/// port exposes [`AudioRecorder::feed`] so a caller (or test) can push int16
/// PCM chunks.  `feed` reproduces the exact callback logic: RMS / peak
/// tracking, frame collection, and the dip-tolerance silence state machine,
/// returning `true` when the silence callback should fire.
pub struct AudioRecorder {
    inner: Mutex<RecorderState>,
    /// Fires once when silence (or no-speech timeout) is detected.
    on_silence_stop: Mutex<Option<Box<dyn FnMut() + Send>>>,
}

struct RecorderState {
    frames: Vec<i16>,
    recording: bool,
    start_time: Option<Instant>,

    // Silence detection state
    has_spoken: bool,
    speech_start: Option<Instant>,
    dip_start: Option<Instant>,
    silence_start: Option<Instant>,
    resume_start: Option<Instant>,
    resume_dip_start: Option<Instant>,

    min_speech_duration: f64,
    max_dip_tolerance: f64,
    silence_threshold: i32,
    silence_duration: f64,
    max_wait: f64,

    peak_rms: i32,
    current_rms: i32,
    silence_callback_armed: bool,
}

impl Default for RecorderState {
    fn default() -> Self {
        RecorderState {
            frames: Vec::new(),
            recording: false,
            start_time: None,
            has_spoken: false,
            speech_start: None,
            dip_start: None,
            silence_start: None,
            resume_start: None,
            resume_dip_start: None,
            min_speech_duration: 0.3,
            max_dip_tolerance: 0.3,
            silence_threshold: SILENCE_RMS_THRESHOLD,
            silence_duration: SILENCE_DURATION_SECONDS,
            max_wait: 15.0,
            peak_rms: 0,
            current_rms: 0,
            silence_callback_armed: false,
        }
    }
}

impl Default for AudioRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioRecorder {
    pub const SUPPORTS_SILENCE_AUTOSTOP: bool = true;

    pub fn new() -> Self {
        AudioRecorder {
            inner: Mutex::new(RecorderState::default()),
            on_silence_stop: Mutex::new(None),
        }
    }

    pub fn elapsed_seconds(&self) -> f64 {
        let st = self.inner.lock().unwrap();
        if !st.recording {
            return 0.0;
        }
        st.start_time.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0)
    }

    /// Current audio input RMS level (0-32767). Updated each audio chunk.
    pub fn current_rms(&self) -> i32 {
        self.inner.lock().unwrap().current_rms
    }

    /// Whether audio recording is currently active.
    pub fn is_recording(&self) -> bool {
        self.inner.lock().unwrap().recording
    }

    /// Start capturing audio.  Resets all detection state.  An optional
    /// silence callback fires once when silence is detected after speech (or
    /// when no speech occurs within `max_wait`).
    pub fn start(&self, on_silence_stop: Option<Box<dyn FnMut() + Send>>) {
        {
            let mut st = self.inner.lock().unwrap();
            if st.recording {
                return; // already recording
            }
            let defaults = RecorderState::default();
            st.frames.clear();
            st.start_time = Some(Instant::now());
            st.has_spoken = false;
            st.speech_start = None;
            st.dip_start = None;
            st.silence_start = None;
            st.resume_start = None;
            st.resume_dip_start = None;
            st.peak_rms = 0;
            st.current_rms = 0;
            st.silence_callback_armed = on_silence_stop.is_some();
            // keep configured thresholds
            st.min_speech_duration = defaults.min_speech_duration;
            st.max_dip_tolerance = defaults.max_dip_tolerance;
            st.silence_threshold = defaults.silence_threshold;
            st.silence_duration = defaults.silence_duration;
            st.max_wait = defaults.max_wait;
            st.recording = true;
        }
        *self.on_silence_stop.lock().unwrap() = on_silence_stop;
        log::info!(
            "Voice recording started (rate={}, channels={})",
            SAMPLE_RATE,
            CHANNELS
        );
    }

    /// Feed an int16 PCM chunk into the recorder, as the PortAudio callback
    /// would.  Updates RMS/peak, collects frames, and runs the silence state
    /// machine.  If silence (or no-speech timeout) is detected, the armed
    /// `on_silence_stop` callback fires exactly once.
    pub fn feed(&self, chunk: &[i16]) {
        let fired;
        {
            let mut st = self.inner.lock().unwrap();
            if !st.recording {
                return; // stream idle -- discard
            }
            st.frames.extend_from_slice(chunk);

            let rms = rms_i16(chunk);
            st.current_rms = rms;
            if rms > st.peak_rms {
                st.peak_rms = rms;
            }

            if !st.silence_callback_armed {
                return;
            }
            fired = st.update_silence(rms, Instant::now());
        }
        if fired {
            self.fire_silence_callback();
        }
    }

    fn fire_silence_callback(&self) {
        let cb = {
            let mut guard = self.on_silence_stop.lock().unwrap();
            guard.take()
        };
        if let Some(mut cb) = cb {
            // Run on a daemon thread, mirroring the Python `_safe_cb` wrapper.
            std::thread::spawn(move || {
                cb();
            });
        }
    }

    /// Stop recording and write captured audio to a WAV file, applying the
    /// short-recording and too-quiet guards.  Returns the path, or `None`.
    pub fn stop(&self) -> Option<PathBuf> {
        let (frames, peak_rms, elapsed) = {
            let mut st = self.inner.lock().unwrap();
            if !st.recording {
                return None;
            }
            st.recording = false;
            st.current_rms = 0;
            if st.frames.is_empty() {
                return None;
            }
            let elapsed = st.start_time.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0);
            let frames = std::mem::take(&mut st.frames);
            (frames, st.peak_rms, elapsed)
        };

        log::info!(
            "Voice recording stopped ({:.1}s, {} samples)",
            elapsed,
            frames.len()
        );

        // Skip very short recordings (< 0.3s of audio).
        let min_samples = (SAMPLE_RATE as f64 * 0.3) as usize;
        if frames.len() < min_samples {
            log::debug!("Recording too short ({} samples), discarding", frames.len());
            return None;
        }

        // Skip silent recordings using peak RMS.
        if peak_rms < SILENCE_RMS_THRESHOLD {
            log::info!(
                "Recording too quiet (peak RMS={} < {}), discarding",
                peak_rms,
                SILENCE_RMS_THRESHOLD
            );
            return None;
        }

        match write_wav(&frames) {
            Ok(p) => Some(p),
            Err(e) => {
                log::error!("WAV write failed: {e}");
                None
            }
        }
    }

    /// Stop recording and discard all captured audio.
    pub fn cancel(&self) {
        {
            let mut st = self.inner.lock().unwrap();
            st.recording = false;
            st.frames.clear();
            st.silence_callback_armed = false;
            st.current_rms = 0;
        }
        *self.on_silence_stop.lock().unwrap() = None;
        log::info!("Voice recording cancelled");
    }

    /// Release the recorder.  (No persistent OS stream in this port.)
    pub fn shutdown(&self) {
        {
            let mut st = self.inner.lock().unwrap();
            st.recording = false;
            st.frames.clear();
            st.silence_callback_armed = false;
        }
        *self.on_silence_stop.lock().unwrap() = None;
        log::info!("AudioRecorder shut down");
    }
}

impl RecorderState {
    /// One iteration of the Python callback's silence-detection block.
    /// Returns `true` if the silence callback should fire.
    fn update_silence(&mut self, rms: i32, now: Instant) -> bool {
        let start = match self.start_time {
            Some(t) => t,
            None => return false,
        };
        let elapsed = now.duration_since(start).as_secs_f64();
        let secs = |from: Option<Instant>| -> f64 {
            from.map(|t| now.duration_since(t).as_secs_f64()).unwrap_or(0.0)
        };

        if rms > self.silence_threshold {
            // Above threshold -- speech (or noise).
            self.dip_start = None;
            if self.speech_start.is_none() {
                self.speech_start = Some(now);
            } else if !self.has_spoken && secs(self.speech_start) >= self.min_speech_duration {
                self.has_spoken = true;
                log::debug!("Speech confirmed ({:.2}s above threshold)", secs(self.speech_start));
            }
            if !self.has_spoken {
                self.silence_start = None;
            } else {
                // Track resumed speech with dip tolerance.
                self.resume_dip_start = None;
                if self.resume_start.is_none() {
                    self.resume_start = Some(now);
                } else if secs(self.resume_start) >= self.min_speech_duration {
                    self.silence_start = None;
                    self.resume_start = None;
                }
            }
        } else if self.has_spoken {
            // Below threshold after speech confirmed.
            if self.resume_start.is_some() {
                if self.resume_dip_start.is_none() {
                    self.resume_dip_start = Some(now);
                } else if secs(self.resume_dip_start) >= self.max_dip_tolerance {
                    self.resume_start = None;
                    self.resume_dip_start = None;
                }
            }
        } else if self.speech_start.is_some() {
            // Speech attempt but RMS dipped -- tolerate brief dips.
            if self.dip_start.is_none() {
                self.dip_start = Some(now);
            } else if secs(self.dip_start) >= self.max_dip_tolerance {
                log::debug!("Speech attempt reset (dip lasted {:.2}s)", secs(self.dip_start));
                self.speech_start = None;
                self.dip_start = None;
            }
        }

        // Fire silence callback when:
        //   1. spoke then went silent for silence_duration, OR
        //   2. no speech detected at all for max_wait seconds.
        let mut should_fire = false;
        if self.has_spoken && rms <= self.silence_threshold {
            if self.silence_start.is_none() {
                self.silence_start = Some(now);
            } else if secs(self.silence_start) >= self.silence_duration {
                log::info!("Silence detected ({:.1}s), auto-stopping", self.silence_duration);
                should_fire = true;
            }
        } else if !self.has_spoken && elapsed >= self.max_wait {
            log::info!("No speech within {:.0}s, auto-stopping", self.max_wait);
            should_fire = true;
        }

        if should_fire {
            // Disarm so it fires only once.
            self.silence_callback_armed = false;
        }
        should_fire
    }
}

impl Recorder for AudioRecorder {
    fn is_recording(&self) -> bool {
        AudioRecorder::is_recording(self)
    }
    fn elapsed_seconds(&self) -> f64 {
        AudioRecorder::elapsed_seconds(self)
    }
    fn current_rms(&self) -> i32 {
        AudioRecorder::current_rms(self)
    }
    fn supports_silence_autostop(&self) -> bool {
        Self::SUPPORTS_SILENCE_AUTOSTOP
    }
}

// ---------------------------------------------------------------------------
// TermuxAudioRecorder
// ---------------------------------------------------------------------------

/// Recorder backend that uses Termux:API microphone capture commands.
pub struct TermuxAudioRecorder {
    inner: Mutex<TermuxState>,
}

#[derive(Default)]
struct TermuxState {
    recording: bool,
    start_time: Option<Instant>,
    recording_path: Option<PathBuf>,
    current_rms: i32,
}

impl Default for TermuxAudioRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl TermuxAudioRecorder {
    pub const SUPPORTS_SILENCE_AUTOSTOP: bool = false;

    pub fn new() -> Self {
        TermuxAudioRecorder {
            inner: Mutex::new(TermuxState::default()),
        }
    }

    pub fn is_recording(&self) -> bool {
        self.inner.lock().unwrap().recording
    }

    pub fn elapsed_seconds(&self) -> f64 {
        let st = self.inner.lock().unwrap();
        if !st.recording {
            return 0.0;
        }
        st.start_time.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0)
    }

    pub fn current_rms(&self) -> i32 {
        self.inner.lock().unwrap().current_rms
    }

    /// Start a Termux:API microphone recording.  Returns an error string when
    /// the Termux API package/app is missing or the start command fails.
    pub fn start(&self) -> Result<(), String> {
        let mic_cmd = termux_microphone_command().ok_or_else(|| {
            "Termux voice capture requires the termux-api package and app.\n\
             Install with: pkg install termux-api\n\
             Then install/update the Termux:API Android app."
                .to_string()
        })?;
        if !termux_api_app_installed() {
            return Err("Termux voice capture requires the Termux:API Android app.\n\
                 Install/update the Termux:API app, then retry /voice on."
                .to_string());
        }

        let path;
        {
            let mut st = self.inner.lock().unwrap();
            if st.recording {
                return Ok(());
            }
            if let Err(e) = fs::create_dir_all(temp_dir()) {
                return Err(format!("Termux microphone start failed: {e}"));
            }
            let timestamp = timestamp_now();
            path = temp_dir().join(format!("recording_{timestamp}.aac"));
            st.recording_path = Some(path.clone());
        }

        let output = Command::new(&mic_cmd)
            .args([
                "-f",
                &path.to_string_lossy(),
                "-l",
                "0",
                "-e",
                "aac",
                "-r",
                &SAMPLE_RATE.to_string(),
                "-c",
                &CHANNELS.to_string(),
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();

        match output {
            Ok(out) if out.status.success() => {}
            Ok(out) => {
                let details = {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let s = if !stderr.trim().is_empty() {
                        stderr.trim().to_string()
                    } else {
                        stdout.trim().to_string()
                    };
                    s
                };
                return Err(format!("Termux microphone start failed: {details}"));
            }
            Err(e) => return Err(format!("Termux microphone start failed: {e}")),
        }

        {
            let mut st = self.inner.lock().unwrap();
            st.start_time = Some(Instant::now());
            st.recording = true;
            st.current_rms = 0;
        }
        log::info!("Termux voice recording started");
        Ok(())
    }

    fn stop_termux_recording(&self) {
        if let Some(mic_cmd) = termux_microphone_command() {
            let _ = Command::new(&mic_cmd)
                .arg("-q")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output();
        }
    }

    /// Stop recording.  Returns the recorded file path, or `None` if the
    /// recording was too short / empty (those files are removed).
    pub fn stop(&self) -> Option<PathBuf> {
        let (path, started_at) = {
            let mut st = self.inner.lock().unwrap();
            if !st.recording {
                return None;
            }
            st.recording = false;
            let path = st.recording_path.take();
            let started = st.start_time;
            st.current_rms = 0;
            (path, started)
        };

        self.stop_termux_recording();

        let path = path?;
        if !path.is_file() {
            return None;
        }
        let elapsed = started_at.map(|t| t.elapsed().as_secs_f64()).unwrap_or(f64::INFINITY);
        if elapsed < 0.3 {
            let _ = fs::remove_file(&path);
            return None;
        }
        let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        if size == 0 {
            let _ = fs::remove_file(&path);
            return None;
        }
        log::info!("Termux voice recording stopped: {}", path.display());
        Some(path)
    }

    /// Stop and discard the recording.
    pub fn cancel(&self) {
        let path = {
            let mut st = self.inner.lock().unwrap();
            let path = st.recording_path.take();
            st.recording = false;
            st.current_rms = 0;
            path
        };
        self.stop_termux_recording();
        if let Some(path) = path {
            if path.is_file() {
                let _ = fs::remove_file(&path);
            }
        }
        log::info!("Termux voice recording cancelled");
    }

    pub fn shutdown(&self) {
        self.cancel();
    }
}

impl Recorder for TermuxAudioRecorder {
    fn is_recording(&self) -> bool {
        TermuxAudioRecorder::is_recording(self)
    }
    fn elapsed_seconds(&self) -> f64 {
        TermuxAudioRecorder::elapsed_seconds(self)
    }
    fn current_rms(&self) -> i32 {
        TermuxAudioRecorder::current_rms(self)
    }
    fn supports_silence_autostop(&self) -> bool {
        Self::SUPPORTS_SILENCE_AUTOSTOP
    }
}

/// The backend chosen by [`create_audio_recorder`].
pub enum RecorderBackend {
    Audio(AudioRecorder),
    Termux(TermuxAudioRecorder),
}

/// Return the best recorder backend for the current environment.
pub fn create_audio_recorder() -> RecorderBackend {
    if termux_voice_capture_available() {
        RecorderBackend::Termux(TermuxAudioRecorder::new())
    } else {
        RecorderBackend::Audio(AudioRecorder::new())
    }
}

// ---------------------------------------------------------------------------
// Audio playback (interruptable)
// ---------------------------------------------------------------------------

/// Handle to the active system playback process so it can be interrupted.
static ACTIVE_PLAYBACK: Mutex<Option<Arc<Mutex<std::process::Child>>>> = Mutex::new(None);

/// Interrupt the currently playing audio (if any).
pub fn stop_playback() {
    let proc = {
        let mut guard = ACTIVE_PLAYBACK.lock().unwrap();
        guard.take()
    };
    if let Some(proc) = proc {
        let mut child = proc.lock().unwrap();
        // Only terminate if still running.
        if matches!(child.try_wait(), Ok(None)) {
            if child.kill().is_ok() {
                log::info!("Audio playback interrupted");
            }
        }
    }
    // (No sounddevice backend to stop in this port.)
}

/// Play an audio file through a system audio player.
///
/// Player order mirrors the Python fallback: `afplay` (macOS only) →
/// `ffplay` (cross-platform) → `aplay` (Linux only).  Playback can be
/// interrupted via [`stop_playback`].  Returns `true` on success.
pub fn play_audio_file(file_path: &str) -> bool {
    let path = Path::new(file_path);
    if !path.is_file() {
        log::warn!("Audio file not found: {file_path}");
        return false;
    }

    // (WAV-via-sounddevice path from Python is unavailable in this port; we go
    // straight to the system-player fallback.)
    let mut players: Vec<Vec<String>> = Vec::new();
    if cfg!(target_os = "macos") {
        players.push(vec!["afplay".into(), file_path.into()]);
    }
    players.push(vec![
        "ffplay".into(),
        "-nodisp".into(),
        "-autoexit".into(),
        "-loglevel".into(),
        "quiet".into(),
        file_path.into(),
    ]);
    if cfg!(target_os = "linux") {
        players.push(vec!["aplay".into(), "-q".into(), file_path.into()]);
    }

    for cmd in &players {
        if which(&cmd[0]).is_none() {
            continue;
        }
        let spawned = Command::new(&cmd[0])
            .args(&cmd[1..])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
        let child = match spawned {
            Ok(c) => c,
            Err(e) => {
                log::debug!("System player {} failed: {e}", cmd[0]);
                continue;
            }
        };
        let child = Arc::new(Mutex::new(child));
        {
            let mut guard = ACTIVE_PLAYBACK.lock().unwrap();
            *guard = Some(child.clone());
        }
        let result = wait_with_timeout(&child, Duration::from_secs(300));
        {
            let mut guard = ACTIVE_PLAYBACK.lock().unwrap();
            *guard = None;
        }
        match result {
            WaitResult::Exited => return true,
            WaitResult::TimedOut => {
                log::warn!("System player {} timed out, killing process", cmd[0]);
                let mut c = child.lock().unwrap();
                let _ = c.kill();
                let _ = c.wait();
            }
            WaitResult::Error => {}
        }
    }

    log::warn!("No audio player available for {file_path}");
    false
}

enum WaitResult {
    Exited,
    TimedOut,
    Error,
}

/// Poll-wait for a child process up to `timeout`.
fn wait_with_timeout(child: &Arc<Mutex<std::process::Child>>, timeout: Duration) -> WaitResult {
    let deadline = Instant::now() + timeout;
    loop {
        {
            let mut c = child.lock().unwrap();
            match c.try_wait() {
                Ok(Some(_)) => return WaitResult::Exited,
                Ok(None) => {}
                Err(_) => return WaitResult::Error,
            }
        }
        if Instant::now() >= deadline {
            return WaitResult::TimedOut;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

// ---------------------------------------------------------------------------
// Requirements check
// ---------------------------------------------------------------------------

/// STT provider summary supplied by the caller (transcription tools layer).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub struct SttStatus {
    pub enabled: bool,
    /// "none" | "local" | "groq" | "openai" | ...
    pub provider: String,
}

impl SttStatus {
    /// STT is usable when enabled and a concrete provider is configured.
    pub fn available(&self) -> bool {
        self.enabled && self.provider != "none"
    }
}

/// Result of [`check_voice_requirements`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub struct VoiceRequirements {
    pub available: bool,
    pub audio_available: bool,
    pub stt_available: bool,
    pub missing_packages: Vec<String>,
    pub details: String,
    pub environment: AudioEnvironment,
}

/// Check if all voice mode requirements are met.
///
/// The STT provider state is passed in (the Python version reads it from the
/// transcription-tools config); everything else is computed natively.
pub fn check_voice_requirements(stt: &SttStatus) -> VoiceRequirements {
    let stt_available = stt.available();

    let mut missing: Vec<String> = Vec::new();
    let termux_capture = termux_voice_capture_available();
    let has_audio = audio_available() || termux_capture;

    if !has_audio {
        missing.push("sounddevice".to_string());
        missing.push("numpy".to_string());
    }

    let env_check = detect_audio_environment();
    let available = has_audio && stt_available && env_check.available;

    let mut details_parts: Vec<String> = Vec::new();
    if termux_capture {
        details_parts.push("Audio capture: OK (Termux:API microphone)".to_string());
    } else if has_audio {
        details_parts.push("Audio capture: OK".to_string());
    } else {
        details_parts.push(format!(
            "Audio capture: MISSING ({})",
            voice_capture_install_hint()
        ));
    }

    if !stt.enabled {
        details_parts.push("STT provider: DISABLED in config (stt.enabled: false)".to_string());
    } else if stt.provider == "local" {
        details_parts.push("STT provider: OK (local faster-whisper)".to_string());
    } else if stt.provider == "groq" {
        details_parts.push("STT provider: OK (Groq)".to_string());
    } else if stt.provider == "openai" {
        details_parts.push("STT provider: OK (OpenAI)".to_string());
    } else {
        details_parts.push(
            "STT provider: MISSING (pip install faster-whisper, or set GROQ_API_KEY / \
             VOICE_TOOLS_OPENAI_KEY)"
                .to_string(),
        );
    }

    for warning in &env_check.warnings {
        details_parts.push(format!("Environment: {warning}"));
    }
    for notice in &env_check.notices {
        details_parts.push(format!("Environment: {notice}"));
    }

    VoiceRequirements {
        available,
        audio_available: has_audio,
        stt_available,
        missing_packages: missing,
        details: details_parts.join("\n"),
        environment: env_check,
    }
}

// ---------------------------------------------------------------------------
// Temp file cleanup
// ---------------------------------------------------------------------------

/// Remove old temporary voice recording files (`recording_*.wav`) older than
/// `max_age_seconds`.  Returns the number of files deleted.
pub fn cleanup_temp_recordings(max_age_seconds: u64) -> usize {
    let dir = temp_dir();
    if !dir.is_dir() {
        return 0;
    }

    let mut deleted = 0usize;
    let now = SystemTime::now();

    let entries = match fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("recording_") || !name.ends_with(".wav") {
            continue;
        }
        let meta = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        if !meta.is_file() {
            continue;
        }
        let modified = match meta.modified() {
            Ok(m) => m,
            Err(_) => continue,
        };
        let age = now.duration_since(modified).unwrap_or(Duration::ZERO).as_secs();
        if age > max_age_seconds {
            if fs::remove_file(entry.path()).is_ok() {
                deleted += 1;
            }
        }
    }

    if deleted > 0 {
        log::debug!("Cleaned up {deleted} old voice recordings");
    }
    deleted
}

/// Default cleanup age (1 hour), matching the Python default argument.
pub const DEFAULT_CLEANUP_MAX_AGE_SECONDS: u64 = 3600;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn epoch() -> Instant {
        Instant::now()
    }

    #[test]
    fn hallucination_exact_and_punctuation() {
        assert!(is_whisper_hallucination("Thank you."));
        assert!(is_whisper_hallucination("thank you"));
        assert!(is_whisper_hallucination("  BYE!  "));
        assert!(is_whisper_hallucination("the end."));
        assert!(is_whisper_hallucination("amara.org"));
        // Non-English
        assert!(is_whisper_hallucination("продолжение следует"));
        assert!(is_whisper_hallucination("ご視聴ありがとうございました"));
    }

    #[test]
    fn hallucination_empty_is_filtered() {
        assert!(is_whisper_hallucination(""));
        assert!(is_whisper_hallucination("   "));
    }

    #[test]
    fn hallucination_repeat_pattern() {
        assert!(is_whisper_hallucination("Thank you. Thank you. Thank you. you"));
        assert!(is_whisper_hallucination("ok okay ok"));
        assert!(is_whisper_hallucination("bye bye bye"));
    }

    #[test]
    fn real_speech_not_filtered() {
        assert!(!is_whisper_hallucination("hello there how are you"));
        assert!(!is_whisper_hallucination("write a function in rust"));
        assert!(!is_whisper_hallucination("thank you for the great work today"));
    }

    #[test]
    fn apply_filter_blanks_hallucination() {
        let r = TranscriptionResult {
            success: true,
            transcript: "thank you".into(),
            error: None,
            filtered: false,
        };
        let out = apply_hallucination_filter(r);
        assert!(out.success);
        assert!(out.filtered);
        assert_eq!(out.transcript, "");
    }

    #[test]
    fn apply_filter_keeps_real_transcript() {
        let r = TranscriptionResult {
            success: true,
            transcript: "list my files".into(),
            error: None,
            filtered: false,
        };
        let out = apply_hallucination_filter(r.clone());
        assert_eq!(out, r);
    }

    #[test]
    fn apply_filter_ignores_failures() {
        let r = TranscriptionResult {
            success: false,
            transcript: "thank you".into(),
            error: Some("boom".into()),
            filtered: false,
        };
        let out = apply_hallucination_filter(r.clone());
        assert_eq!(out, r);
    }

    #[test]
    fn rms_of_silence_is_zero() {
        assert_eq!(rms_i16(&[0i16; 100]), 0);
        assert_eq!(rms_i16(&[]), 0);
    }

    #[test]
    fn rms_of_constant_signal() {
        // RMS of a constant +1000 signal is 1000.
        assert_eq!(rms_i16(&[1000i16; 50]), 1000);
    }

    #[test]
    fn wav_header_is_well_formed() {
        let dir = std::env::temp_dir().join(format!("hermes_voice_test_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.wav");
        let samples: Vec<i16> = (0..100).map(|i| (i as i16) * 10).collect();
        write_wav_to(&path, &samples, SAMPLE_RATE, CHANNELS).unwrap();

        let bytes = fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[12..16], b"fmt ");
        assert_eq!(&bytes[36..40], b"data");
        // data length = samples * 2
        let data_len = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
        assert_eq!(data_len as usize, samples.len() * 2);
        // sample rate
        let sr = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
        assert_eq!(sr, SAMPLE_RATE);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn recorder_lifecycle_flags() {
        let rec = AudioRecorder::new();
        assert!(!rec.is_recording());
        assert_eq!(rec.elapsed_seconds(), 0.0);
        rec.start(None);
        assert!(rec.is_recording());
        rec.feed(&[5000i16; 1600]);
        assert!(rec.current_rms() > SILENCE_RMS_THRESHOLD);
        rec.cancel();
        assert!(!rec.is_recording());
        assert_eq!(rec.current_rms(), 0);
    }

    #[test]
    fn recorder_discards_short_recording() {
        let rec = AudioRecorder::new();
        rec.start(None);
        // Fewer than 0.3s of samples (< 4800).
        rec.feed(&[5000i16; 100]);
        assert!(rec.stop().is_none());
    }

    #[test]
    fn recorder_discards_quiet_recording() {
        let rec = AudioRecorder::new();
        rec.start(None);
        // Enough samples, but all below the silence threshold.
        rec.feed(&[50i16; 8000]);
        assert!(rec.stop().is_none());
    }

    #[test]
    fn recorder_writes_loud_recording() {
        let rec = AudioRecorder::new();
        rec.start(None);
        rec.feed(&[8000i16; 8000]);
        let path = rec.stop();
        assert!(path.is_some());
        let p = path.unwrap();
        assert!(p.exists());
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn silence_state_machine_fires_after_speech_then_silence() {
        let mut st = RecorderState::default();
        st.silence_callback_armed = true;
        let start = epoch();
        st.start_time = Some(start);

        // Speak loudly for 0.4s to confirm speech.
        let mut t = start;
        let mut fired = false;
        for _ in 0..5 {
            t += Duration::from_millis(100);
            fired |= st.update_silence(5000, t);
        }
        assert!(st.has_spoken, "speech should be confirmed after >0.3s");
        assert!(!fired);

        // Now go silent for > silence_duration (3s).
        for _ in 0..40 {
            t += Duration::from_millis(100);
            fired |= st.update_silence(0, t);
        }
        assert!(fired, "silence callback should fire after sustained silence");
        assert!(!st.silence_callback_armed, "callback disarms after firing");
    }

    #[test]
    fn silence_state_machine_fires_on_no_speech_timeout() {
        let mut st = RecorderState::default();
        st.silence_callback_armed = true;
        let start = epoch();
        st.start_time = Some(start);

        let mut t = start;
        let mut fired = false;
        // Stay quiet past max_wait (15s).
        for _ in 0..160 {
            t += Duration::from_millis(100);
            fired |= st.update_silence(10, t);
        }
        assert!(fired, "should auto-stop after max_wait with no speech");
        assert!(!st.has_spoken);
    }

    #[test]
    fn silence_state_machine_tolerates_dips() {
        let mut st = RecorderState::default();
        st.silence_callback_armed = true;
        let start = epoch();
        st.start_time = Some(start);

        let mut t = start;
        // Confirm speech.
        for _ in 0..5 {
            t += Duration::from_millis(100);
            st.update_silence(5000, t);
        }
        assert!(st.has_spoken);

        // Brief dip (< 3s) then speech resumes -- should not fire.
        let mut fired = false;
        for _ in 0..10 {
            t += Duration::from_millis(100);
            fired |= st.update_silence(0, t);
        }
        // 1s of silence < 3s threshold.
        assert!(!fired);
        // Resume speaking.
        for _ in 0..5 {
            t += Duration::from_millis(100);
            fired |= st.update_silence(5000, t);
        }
        assert!(!fired);
    }

    #[test]
    fn env_detection_returns_struct() {
        let env = detect_audio_environment();
        // available iff there are no warnings.
        assert_eq!(env.available, env.warnings.is_empty());
    }

    #[test]
    fn termux_backend_selected_only_when_available() {
        // Outside Termux, must pick the AudioRecorder backend.
        if !termux_voice_capture_available() {
            match create_audio_recorder() {
                RecorderBackend::Audio(_) => {}
                RecorderBackend::Termux(_) => panic!("should not pick Termux outside Termux"),
            }
        }
    }

    #[test]
    fn stt_status_availability() {
        assert!(SttStatus { enabled: true, provider: "groq".into() }.available());
        assert!(!SttStatus { enabled: false, provider: "groq".into() }.available());
        assert!(!SttStatus { enabled: true, provider: "none".into() }.available());
    }

    #[test]
    fn requirements_details_format() {
        let stt = SttStatus { enabled: true, provider: "local".into() };
        let req = check_voice_requirements(&stt);
        assert!(req.stt_available);
        assert!(req.details.contains("STT provider: OK (local faster-whisper)"));
    }

    #[test]
    fn cleanup_missing_dir_returns_zero() {
        // Hard to guarantee absence of the real dir; just assert it runs.
        let _ = cleanup_temp_recordings(DEFAULT_CLEANUP_MAX_AGE_SECONDS);
    }

    #[test]
    fn play_missing_file_returns_false() {
        assert!(!play_audio_file("/nonexistent/path/to/file.wav"));
    }
}
