//! AES-256-GCM utilities for QQBot scan-to-configure credential decryption.
//!
//! Faithful Rust port of `gateway/platforms/qqbot/crypto.py`.
//!
//! The flow is: this side generates a random 256-bit AES key (base64) which is
//! handed to the server via `create_bind_task`. The server encrypts the bot's
//! `client_secret` with AES-256-GCM and returns it base64-encoded. This module
//! decrypts that ciphertext.
//!
//! Ciphertext layout (after base64-decoding):
//!
//! ```text
//! IV (12 bytes) ‖ ciphertext (N bytes) ‖ AuthTag (16 bytes)
//! ```

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine as _;

/// Length of the GCM nonce / IV in bytes.
pub const IV_LEN: usize = 12;
/// Length of the GCM authentication tag in bytes.
pub const TAG_LEN: usize = 16;
/// Length of the AES-256 key in bytes.
pub const KEY_LEN: usize = 32;

/// Errors that can occur while generating keys or decrypting secrets.
#[derive(Debug)]
pub enum QqCryptoError {
    /// A base64 input could not be decoded.
    Base64(base64::DecodeError),
    /// The decoded key was not 32 bytes (required for AES-256).
    InvalidKeyLength(usize),
    /// The decoded ciphertext was too short to contain IV + tag.
    CiphertextTooShort(usize),
    /// AES-GCM decryption / authentication failed.
    Decrypt,
    /// The decrypted plaintext was not valid UTF-8.
    InvalidUtf8(std::string::FromUtf8Error),
}

impl std::fmt::Display for QqCryptoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QqCryptoError::Base64(e) => write!(f, "base64 decode error: {e}"),
            QqCryptoError::InvalidKeyLength(n) => {
                write!(f, "invalid AES key length: expected {KEY_LEN} bytes, got {n}")
            }
            QqCryptoError::CiphertextTooShort(n) => write!(
                f,
                "ciphertext too short: need at least {} bytes, got {n}",
                IV_LEN + TAG_LEN
            ),
            QqCryptoError::Decrypt => write!(f, "AES-GCM decryption/authentication failed"),
            QqCryptoError::InvalidUtf8(e) => write!(f, "decrypted plaintext is not valid UTF-8: {e}"),
        }
    }
}

impl std::error::Error for QqCryptoError {}

impl From<base64::DecodeError> for QqCryptoError {
    fn from(e: base64::DecodeError) -> Self {
        QqCryptoError::Base64(e)
    }
}

/// Generate a 256-bit random AES key and return it as standard base64.
///
/// Mirrors `generate_bind_key()`: 32 cryptographically-random bytes,
/// base64-encoded. The key is passed to `create_bind_task` so the server can
/// encrypt the bot's `client_secret` before returning it. Only this side holds
/// the key, ensuring the secret never travels in plaintext.
pub fn generate_bind_key() -> String {
    let mut key = [0u8; KEY_LEN];
    getrandom::fill(&mut key).expect("failed to read OS randomness for AES key");
    B64.encode(key)
}

/// Decrypt a base64-encoded AES-256-GCM ciphertext.
///
/// Mirrors `decrypt_secret()`.
///
/// * `encrypted_base64` — the `bot_encrypt_secret` value from
///   `poll_bind_result`.
/// * `key_base64` — the base64 AES key produced by [`generate_bind_key`].
///
/// Returns the decrypted `client_secret` as a UTF-8 string.
pub fn decrypt_secret(encrypted_base64: &str, key_base64: &str) -> Result<String, QqCryptoError> {
    let key_bytes = B64.decode(key_base64.trim())?;
    if key_bytes.len() != KEY_LEN {
        return Err(QqCryptoError::InvalidKeyLength(key_bytes.len()));
    }

    let raw = B64.decode(encrypted_base64.trim())?;
    if raw.len() < IV_LEN + TAG_LEN {
        return Err(QqCryptoError::CiphertextTooShort(raw.len()));
    }

    let iv = &raw[..IV_LEN];
    // Python passes `ciphertext + tag` to AESGCM.decrypt; the Rust `aes-gcm`
    // crate's `decrypt` likewise expects the tag appended to the ciphertext.
    let ciphertext_with_tag = &raw[IV_LEN..];

    let key = Key::<Aes256Gcm>::from_slice(&key_bytes);
    let cipher = Aes256Gcm::new(key);
    let nonce = Nonce::from_slice(iv);

    let plaintext = cipher
        .decrypt(nonce, ciphertext_with_tag)
        .map_err(|_| QqCryptoError::Decrypt)?;

    String::from_utf8(plaintext).map_err(QqCryptoError::InvalidUtf8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aes_gcm::aead::Aead;

    /// Helper: encrypt a plaintext the same way the QQ server does, producing
    /// the base64 layout `IV ‖ ciphertext ‖ tag` that `decrypt_secret` expects.
    fn encrypt(plaintext: &[u8], key_b64: &str, iv: &[u8; IV_LEN]) -> String {
        let key_bytes = B64.decode(key_b64).unwrap();
        let key = Key::<Aes256Gcm>::from_slice(&key_bytes);
        let cipher = Aes256Gcm::new(key);
        let nonce = Nonce::from_slice(iv);
        let ct_with_tag = cipher.encrypt(nonce, plaintext).unwrap();
        let mut raw = Vec::with_capacity(IV_LEN + ct_with_tag.len());
        raw.extend_from_slice(iv);
        raw.extend_from_slice(&ct_with_tag);
        B64.encode(raw)
    }

    #[test]
    fn generated_key_is_32_bytes_base64() {
        let k = generate_bind_key();
        let decoded = B64.decode(&k).unwrap();
        assert_eq!(decoded.len(), KEY_LEN);
    }

    #[test]
    fn generated_keys_are_unique() {
        let a = generate_bind_key();
        let b = generate_bind_key();
        assert_ne!(a, b);
    }

    #[test]
    fn round_trip_decrypts_to_original() {
        let key_b64 = generate_bind_key();
        let iv = [7u8; IV_LEN];
        let secret = "super-secret-client-secret-Ω";
        let enc = encrypt(secret.as_bytes(), &key_b64, &iv);
        let out = decrypt_secret(&enc, &key_b64).unwrap();
        assert_eq!(out, secret);
    }

    #[test]
    fn empty_plaintext_round_trips() {
        let key_b64 = generate_bind_key();
        let iv = [0u8; IV_LEN];
        let enc = encrypt(b"", &key_b64, &iv);
        let out = decrypt_secret(&enc, &key_b64).unwrap();
        assert_eq!(out, "");
    }

    #[test]
    fn wrong_key_fails_authentication() {
        let key_b64 = generate_bind_key();
        let other_b64 = generate_bind_key();
        let iv = [1u8; IV_LEN];
        let enc = encrypt(b"hello", &key_b64, &iv);
        let err = decrypt_secret(&enc, &other_b64).unwrap_err();
        assert!(matches!(err, QqCryptoError::Decrypt));
    }

    #[test]
    fn tampered_ciphertext_fails() {
        let key_b64 = generate_bind_key();
        let iv = [2u8; IV_LEN];
        let enc = encrypt(b"hello world", &key_b64, &iv);
        let mut raw = B64.decode(&enc).unwrap();
        // flip a bit in the ciphertext body
        let idx = IV_LEN + 1;
        raw[idx] ^= 0x01;
        let tampered = B64.encode(raw);
        let err = decrypt_secret(&tampered, &key_b64).unwrap_err();
        assert!(matches!(err, QqCryptoError::Decrypt));
    }

    #[test]
    fn bad_key_length_rejected() {
        // 16-byte key -> AES-128 size, not allowed for AES-256.
        let short_key = B64.encode([0u8; 16]);
        let err = decrypt_secret(&B64.encode([0u8; 40]), &short_key).unwrap_err();
        assert!(matches!(err, QqCryptoError::InvalidKeyLength(16)));
    }

    #[test]
    fn too_short_ciphertext_rejected() {
        let key_b64 = generate_bind_key();
        // 20 bytes < IV_LEN + TAG_LEN (28)
        let err = decrypt_secret(&B64.encode([0u8; 20]), &key_b64).unwrap_err();
        assert!(matches!(err, QqCryptoError::CiphertextTooShort(20)));
    }

    #[test]
    fn invalid_base64_rejected() {
        let key_b64 = generate_bind_key();
        let err = decrypt_secret("not!!base64", &key_b64).unwrap_err();
        assert!(matches!(err, QqCryptoError::Base64(_)));
    }
}
