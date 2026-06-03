//! Rendering bridge — routes TUI content through optional renderers.
//!
//! Faithful port of `tui_gateway/render.py`.
//!
//! The Python original imported renderers from `agent.rich_output` at call
//! time. When that module (or a given function) was missing, every helper
//! returned `None`, signalling the TUI to fall back to its own
//! `markdown.tsx` renderer.
//!
//! Native Rust has no `agent.rich_output` Python module, so the equivalent
//! mechanism is a process-global registry of optional renderer callbacks.
//! When no renderer is registered the helpers return `None`, exactly
//! reproducing the "fall back to the TUI's own markdown" behaviour. A host
//! that *does* have rich rendering can install callbacks via
//! [`set_message_renderer`], [`set_diff_renderer`] and
//! [`set_stream_renderer_factory`].

use std::sync::{Arc, RwLock};

/// Default column width used by the Python helpers (`cols: int = 80`).
pub const DEFAULT_COLS: u32 = 80;

/// A renderer callback: takes the raw text and a column width, returns the
/// rendered string. Returning `None` means "rendering failed, fall back" —
/// mirroring the bare `except Exception: return None` in the Python source.
pub type RenderFn = Arc<dyn Fn(&str, u32) -> Option<String> + Send + Sync>;

/// Factory for streaming renderers. Mirrors `StreamingRenderer(cols=...)`.
pub type StreamRendererFactory = Arc<dyn Fn(u32) -> Option<Box<dyn StreamRenderer>> + Send + Sync>;

/// Trait matching the Python `StreamingRenderer` object. Implementors
/// incrementally consume text chunks and produce rendered output.
pub trait StreamRenderer: Send {
    /// Feed a chunk of text; returns any newly renderable output.
    fn feed(&mut self, chunk: &str) -> Option<String>;
    /// Flush any buffered content at end of stream.
    fn finish(&mut self) -> Option<String>;
}

struct Registry {
    message: Option<RenderFn>,
    diff: Option<RenderFn>,
    stream: Option<StreamRendererFactory>,
}

impl Registry {
    const fn empty() -> Self {
        Registry {
            message: None,
            diff: None,
            stream: None,
        }
    }
}

fn registry() -> &'static RwLock<Registry> {
    use std::sync::OnceLock;
    static REG: OnceLock<RwLock<Registry>> = OnceLock::new();
    REG.get_or_init(|| RwLock::new(Registry::empty()))
}

/// Install the message renderer (equivalent of `agent.rich_output.format_response`).
pub fn set_message_renderer<F>(f: F)
where
    F: Fn(&str, u32) -> Option<String> + Send + Sync + 'static,
{
    registry().write().unwrap().message = Some(Arc::new(f));
}

/// Install the diff renderer (equivalent of `agent.rich_output.render_diff`).
pub fn set_diff_renderer<F>(f: F)
where
    F: Fn(&str, u32) -> Option<String> + Send + Sync + 'static,
{
    registry().write().unwrap().diff = Some(Arc::new(f));
}

/// Install the streaming-renderer factory (equivalent of
/// `agent.rich_output.StreamingRenderer`).
pub fn set_stream_renderer_factory<F>(f: F)
where
    F: Fn(u32) -> Option<Box<dyn StreamRenderer>> + Send + Sync + 'static,
{
    registry().write().unwrap().stream = Some(Arc::new(f));
}

/// Clear all registered renderers — restores the "no rich_output" baseline.
/// Primarily useful for tests.
pub fn clear_renderers() {
    let mut reg = registry().write().unwrap();
    *reg = Registry::empty();
}

/// Render a message. Returns `None` when no renderer is installed or the
/// renderer fails, signalling the TUI to use its own markdown renderer.
///
/// Port of `render_message(text, cols=80)`.
pub fn render_message(text: &str, cols: u32) -> Option<String> {
    let f = registry().read().unwrap().message.clone()?;
    f(text, cols)
}

/// Convenience: render a message with [`DEFAULT_COLS`].
pub fn render_message_default(text: &str) -> Option<String> {
    render_message(text, DEFAULT_COLS)
}

/// Render a diff. Returns `None` when no renderer is installed or it fails.
///
/// Port of `render_diff(text, cols=80)`.
pub fn render_diff(text: &str, cols: u32) -> Option<String> {
    let f = registry().read().unwrap().diff.clone()?;
    f(text, cols)
}

/// Convenience: render a diff with [`DEFAULT_COLS`].
pub fn render_diff_default(text: &str) -> Option<String> {
    render_diff(text, DEFAULT_COLS)
}

/// Construct a streaming renderer. Returns `None` when no factory is
/// installed or it fails.
///
/// Port of `make_stream_renderer(cols=80)`.
pub fn make_stream_renderer(cols: u32) -> Option<Box<dyn StreamRenderer>> {
    let f = registry().read().unwrap().stream.clone()?;
    f(cols)
}

/// Convenience: build a streaming renderer with [`DEFAULT_COLS`].
pub fn make_stream_renderer_default() -> Option<Box<dyn StreamRenderer>> {
    make_stream_renderer(DEFAULT_COLS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialise tests: they mutate the process-global registry.
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        static M: Mutex<()> = Mutex::new(());
        M.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn no_renderer_returns_none() {
        let _g = lock();
        clear_renderers();
        assert_eq!(render_message("hi", 80), None);
        assert_eq!(render_diff("- a\n+ b", 80), None);
        assert!(make_stream_renderer(80).is_none());
    }

    #[test]
    fn message_renderer_is_used() {
        let _g = lock();
        clear_renderers();
        set_message_renderer(|text, cols| Some(format!("[{cols}]{text}")));
        assert_eq!(render_message("yo", 42), Some("[42]yo".to_string()));
        assert_eq!(render_message_default("yo"), Some("[80]yo".to_string()));
        clear_renderers();
    }

    #[test]
    fn diff_renderer_is_used_and_independent() {
        let _g = lock();
        clear_renderers();
        set_diff_renderer(|text, _cols| Some(format!("DIFF:{text}")));
        assert_eq!(render_diff("x", 80), Some("DIFF:x".to_string()));
        // message renderer is still unset -> None
        assert_eq!(render_message("x", 80), None);
        clear_renderers();
    }

    #[test]
    fn renderer_failure_propagates_none() {
        let _g = lock();
        clear_renderers();
        set_message_renderer(|_t, _c| None);
        assert_eq!(render_message("anything", 80), None);
        clear_renderers();
    }

    struct EchoStream {
        cols: u32,
        buf: String,
    }
    impl StreamRenderer for EchoStream {
        fn feed(&mut self, chunk: &str) -> Option<String> {
            self.buf.push_str(chunk);
            Some(format!("({}){}", self.cols, chunk))
        }
        fn finish(&mut self) -> Option<String> {
            if self.buf.is_empty() {
                None
            } else {
                Some(std::mem::take(&mut self.buf))
            }
        }
    }

    #[test]
    fn stream_factory_builds_renderer() {
        let _g = lock();
        clear_renderers();
        set_stream_renderer_factory(|cols| {
            Some(Box::new(EchoStream {
                cols,
                buf: String::new(),
            }))
        });
        let mut r = make_stream_renderer(7).expect("factory should yield a renderer");
        assert_eq!(r.feed("ab"), Some("(7)ab".to_string()));
        assert_eq!(r.finish(), Some("ab".to_string()));
        // default cols path
        let mut r2 = make_stream_renderer_default().expect("default factory");
        assert_eq!(r2.feed("z"), Some("(80)z".to_string()));
        clear_renderers();
    }

    #[test]
    fn stream_factory_failure_returns_none() {
        let _g = lock();
        clear_renderers();
        set_stream_renderer_factory(|_cols| None);
        assert!(make_stream_renderer(80).is_none());
        clear_renderers();
    }
}
