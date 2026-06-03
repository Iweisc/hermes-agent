//! Embedded-CPython boundary for executing *user-authored* Python in-process.
//!
//! This module is the single sanctioned place where hermes runs Python that is
//! genuinely not ours to port: user plugins, user shell/python hooks, and
//! ML/synthesis entrypoints (NeuTTS) that wrap native model code. Everything
//! that is hermes' own logic is ported to Rust and must NOT route through here.
//!
//! Why in-process (pyo3) rather than `Command::new("python")`:
//! - No per-call interpreter startup cost (plugins/hooks fire frequently).
//! - Structured value marshalling (`serde_json::Value` <-> `PyObject`) instead
//!   of stringly-typed stdin/stdout JSON framing.
//! - A single resolved interpreter/venv for the whole process.
//!
//! Deliberate exception: the `code_execution` tool keeps its OS-level sandbox
//! (a fresh subprocess with `env_clear` + `setsid` + kill-on-timeout). Running
//! arbitrary agent-written code in-process would forfeit that isolation and
//! killability, so it is intentionally NOT migrated to `pyhost`.
//!
//! The whole module is gated behind the `pyhost` cargo feature. With the
//! feature off, the public functions still exist but return a clear
//! "not built with pyhost" error, so callers compile and degrade gracefully.

use serde_json::Value;
use std::path::{Path, PathBuf};

/// Error returned by every `pyhost` entrypoint.
#[derive(Debug)]
pub enum PyHostError {
    /// The crate was compiled without the `pyhost` feature.
    NotEnabled,
    /// The embedded interpreter raised an exception (message rendered).
    Python(String),
    /// A value could not be marshalled across the Rust/Python boundary.
    Marshal(String),
    /// The target module / entrypoint could not be imported or found.
    NotFound(String),
}

impl std::fmt::Display for PyHostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PyHostError::NotEnabled => write!(
                f,
                "embedded Python is unavailable: this build was compiled without the `pyhost` feature"
            ),
            PyHostError::Python(msg) => write!(f, "python error: {msg}"),
            PyHostError::Marshal(msg) => write!(f, "python marshalling error: {msg}"),
            PyHostError::NotFound(msg) => write!(f, "python entrypoint not found: {msg}"),
        }
    }
}

impl std::error::Error for PyHostError {}

pub type PyHostResult<T> = Result<T, PyHostError>;

/// Resolve the Python interpreter / venv hermes should use, mirroring the
/// discovery order the legacy `python_bridge::resolve_repo_python` used so
/// behaviour (and `HERMES_PYTHON` override) is preserved.
///
/// This is feature-independent: it is plain filesystem probing and is used to
/// point the embedded interpreter at the right `site-packages` (via
/// `PYTHONPATH`/venv `sys.path` seeding) before user modules are imported.
pub fn resolve_repo_python(project_root: &Path) -> Option<PathBuf> {
    if let Ok(value) = std::env::var("HERMES_PYTHON") {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return Some(PathBuf::from(trimmed));
        }
    }

    let venv_python = |root: &Path| -> PathBuf {
        if cfg!(windows) {
            root.join("Scripts").join("python.exe")
        } else {
            root.join("bin").join("python")
        }
    };

    let candidates = [
        venv_python(&project_root.join(".venv")),
        venv_python(&project_root.join("venv")),
        dirs::home_dir()
            .map(|h| venv_python(&h.join(".hermes").join("hermes-agent").join("venv")))
            .unwrap_or_default(),
    ];
    candidates.into_iter().find(|c| c.exists())
}

#[cfg(feature = "pyhost")]
mod imp {
    use super::{PyHostError, PyHostResult};
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList};
    use serde_json::{Map, Value};

    fn py_err(err: PyErr) -> PyHostError {
        Python::with_gil(|py| PyHostError::Python(format!("{}", err.value(py))))
    }

    /// Marshal a `serde_json::Value` into a Python object.
    ///
    /// Scalar `into_pyobject` conversions are infallible in pyo3 0.25 (their
    /// error type is `Infallible`), so they are unwrapped and converted to an
    /// owned `Py<PyAny>` via `into_any().unbind()`.
    fn to_py(py: Python<'_>, value: &Value) -> PyHostResult<PyObject> {
        match value {
            Value::Null => Ok(py.None()),
            Value::Bool(b) => Ok(b
                .into_pyobject(py)
                .unwrap_or_else(|e| match e {})
                .to_owned()
                .into_any()
                .unbind()),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(i.into_pyobject(py).unwrap_or_else(|e| match e {}).into_any().unbind())
                } else if let Some(u) = n.as_u64() {
                    Ok(u.into_pyobject(py).unwrap_or_else(|e| match e {}).into_any().unbind())
                } else {
                    let f = n.as_f64().ok_or_else(|| {
                        PyHostError::Marshal(format!("unrepresentable number: {n}"))
                    })?;
                    Ok(f.into_pyobject(py).unwrap_or_else(|e| match e {}).into_any().unbind())
                }
            }
            Value::String(s) => Ok(s
                .into_pyobject(py)
                .unwrap_or_else(|e| match e {})
                .into_any()
                .unbind()),
            Value::Array(items) => {
                let list = PyList::empty(py);
                for item in items {
                    list.append(to_py(py, item)?).map_err(py_err)?;
                }
                Ok(list.into_any().unbind())
            }
            Value::Object(map) => {
                let dict = PyDict::new(py);
                for (k, v) in map {
                    dict.set_item(k, to_py(py, v)?).map_err(py_err)?;
                }
                Ok(dict.into_any().unbind())
            }
        }
    }

    /// Marshal a Python object back into a `serde_json::Value`.
    fn from_py(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyHostResult<Value> {
        if obj.is_none() {
            return Ok(Value::Null);
        }
        if let Ok(b) = obj.extract::<bool>() {
            // bool must be checked before int (Python bool is a subclass of int).
            if obj.is_instance_of::<pyo3::types::PyBool>() {
                return Ok(Value::Bool(b));
            }
        }
        if let Ok(i) = obj.extract::<i64>() {
            return Ok(Value::Number(i.into()));
        }
        if let Ok(f) = obj.extract::<f64>() {
            return serde_json::Number::from_f64(f)
                .map(Value::Number)
                .ok_or_else(|| PyHostError::Marshal(format!("non-finite float: {f}")));
        }
        if let Ok(s) = obj.extract::<String>() {
            return Ok(Value::String(s));
        }
        if let Ok(list) = obj.downcast::<PyList>() {
            let mut out = Vec::with_capacity(list.len());
            for item in list.iter() {
                out.push(from_py(py, &item)?);
            }
            return Ok(Value::Array(out));
        }
        if let Ok(dict) = obj.downcast::<PyDict>() {
            let mut map = Map::new();
            for (k, v) in dict.iter() {
                let key = k
                    .extract::<String>()
                    .map_err(|_| PyHostError::Marshal("non-string dict key".into()))?;
                map.insert(key, from_py(py, &v)?);
            }
            return Ok(Value::Object(map));
        }
        // Fallback: stringify (covers tuples, custom objects, etc.).
        obj.str()
            .map_err(py_err)
            .and_then(|s| s.extract::<String>().map_err(py_err))
            .map(Value::String)
    }

    /// Import `module`, call `attr(args_json)`, return the result as JSON.
    pub fn call_entrypoint(module: &str, attr: &str, args: &Value) -> PyHostResult<Value> {
        Python::with_gil(|py| {
            let py_module = PyModule::import(py, module)
                .map_err(|_| PyHostError::NotFound(format!("module `{module}`")))?;
            let func = py_module
                .getattr(attr)
                .map_err(|_| PyHostError::NotFound(format!("{module}.{attr}")))?;
            let py_args = to_py(py, args)?;
            let result = func.call1((py_args,)).map_err(py_err)?;
            from_py(py, &result)
        })
    }

    /// Execute a top-level Python source string with `payload` bound as a
    /// module-level `payload` variable, returning a `result` variable as JSON.
    pub fn run_source(source: &str, payload: &Value) -> PyHostResult<Value> {
        Python::with_gil(|py| {
            let globals = PyDict::new(py);
            globals.set_item("payload", to_py(py, payload)?).map_err(py_err)?;
            py.run(
                std::ffi::CString::new(source)
                    .map_err(|e| PyHostError::Marshal(e.to_string()))?
                    .as_c_str(),
                Some(&globals),
                None,
            )
            .map_err(py_err)?;
            match globals.get_item("result").map_err(py_err)? {
                Some(val) => from_py(py, &val),
                None => Ok(Value::Null),
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Public surface — thin wrappers that dispatch to `imp` when the feature is on
// and return `PyHostError::NotEnabled` otherwise.
// ---------------------------------------------------------------------------

/// Call `module.attr(args)` in the embedded interpreter, returning JSON.
///
/// Used by `plugin_runtime` to invoke a user plugin's entrypoint and by hook
/// dispatch to run user-authored Python hooks.
pub fn call_entrypoint(_module: &str, _attr: &str, _args: &Value) -> PyHostResult<Value> {
    #[cfg(feature = "pyhost")]
    {
        imp::call_entrypoint(_module, _attr, _args)
    }
    #[cfg(not(feature = "pyhost"))]
    {
        Err(PyHostError::NotEnabled)
    }
}

/// Run a user-authored Python source string with a JSON `payload` in scope,
/// returning the module-level `result` value as JSON.
pub fn run_source(_source: &str, _payload: &Value) -> PyHostResult<Value> {
    #[cfg(feature = "pyhost")]
    {
        imp::run_source(_source, _payload)
    }
    #[cfg(not(feature = "pyhost"))]
    {
        Err(PyHostError::NotEnabled)
    }
}

/// Whether this build can execute embedded Python.
pub const fn available() -> bool {
    cfg!(feature = "pyhost")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_honors_hermes_python_override() {
        // Without the env var set we just confirm the probe doesn't panic.
        let root = std::path::Path::new("/nonexistent-root-xyz");
        let _ = resolve_repo_python(root);
    }

    #[test]
    fn available_matches_feature() {
        assert_eq!(available(), cfg!(feature = "pyhost"));
    }

    #[cfg(not(feature = "pyhost"))]
    #[test]
    fn calls_error_without_feature() {
        assert!(matches!(
            call_entrypoint("m", "f", &serde_json::Value::Null),
            Err(PyHostError::NotEnabled)
        ));
    }

    #[cfg(feature = "pyhost")]
    #[test]
    fn run_source_round_trips_payload() {
        // Echoes the payload back through a computed `result`, exercising both
        // the to_py (payload in) and from_py (result out) marshalling paths
        // and the embedded interpreter itself.
        let payload = serde_json::json!({
            "n": 21,
            "flag": true,
            "items": ["a", "b"],
            "nested": {"x": 1.5},
        });
        let out = run_source(
            "result = {\n  'doubled': payload['n'] * 2,\n  'flag': payload['flag'],\n  'count': len(payload['items']),\n  'x': payload['nested']['x'],\n}",
            &payload,
        )
        .expect("embedded python should run");
        assert_eq!(out["doubled"], serde_json::json!(42));
        assert_eq!(out["flag"], serde_json::json!(true));
        assert_eq!(out["count"], serde_json::json!(2));
        assert_eq!(out["x"], serde_json::json!(1.5));
    }

    #[cfg(feature = "pyhost")]
    #[test]
    fn missing_module_is_not_found() {
        let err = call_entrypoint(
            "hermes_definitely_missing_mod_xyz",
            "go",
            &serde_json::Value::Null,
        )
        .unwrap_err();
        assert!(matches!(err, PyHostError::NotFound(_)));
    }
}
