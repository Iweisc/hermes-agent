//! WeCom `BizMsgCrypt`-compatible AES-CBC encryption for callback mode.
//!
//! Native Rust port of `gateway/platforms/wecom_crypto.py`.
//!
//! Implements the same wire format as Tencent's official `WXBizMsgCrypt`
//! SDK so that WeCom can verify, encrypt, and decrypt callback payloads.
//!
//! The encrypted payload layout (after AES-256-CBC decryption and PKCS#7
//! unpadding) is:
//!
//! ```text
//! [16-byte random prefix][4-byte big-endian msg length][xml content][receive_id]
//! ```
//!
//! AES uses the 32-byte key derived from `base64decode(encoding_aes_key + "=")`
//! and an IV equal to the first 16 bytes of that key.

use std::fmt;

use aes::cipher::{Array, BlockCipherDecrypt, BlockCipherEncrypt, KeyInit};
use aes::Aes256;
use base64::Engine;
use sha1::{Digest, Sha1};

const BLOCK_SIZE: usize = 16;
/// PKCS#7 block size used by the WeCom wire format (note: 32, not 16).
const PKCS7_BLOCK_SIZE: usize = 32;

/// Errors raised by the WeCom crypto helper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeComCryptoError {
    /// Invalid constructor argument (mirrors Python `ValueError`).
    Value(String),
    /// `msg_signature` did not match the locally computed SHA-1 signature.
    Signature(String),
    /// Decryption / payload parsing failure.
    Decrypt(String),
    /// Encryption failure.
    Encrypt(String),
}

impl fmt::Display for WeComCryptoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeComCryptoError::Value(m) => write!(f, "{m}"),
            WeComCryptoError::Signature(m) => write!(f, "{m}"),
            WeComCryptoError::Decrypt(m) => write!(f, "{m}"),
            WeComCryptoError::Encrypt(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for WeComCryptoError {}

type Result<T> = std::result::Result<T, WeComCryptoError>;

/// PKCS#7 padding helper with a 32-byte block size, matching the Python
/// `PKCS7Encoder`.
pub struct Pkcs7Encoder;

impl Pkcs7Encoder {
    /// Block size used for padding (32 bytes).
    pub const BLOCK_SIZE: usize = PKCS7_BLOCK_SIZE;

    /// Append PKCS#7 padding to `text`.
    pub fn encode(text: &[u8]) -> Vec<u8> {
        let mut amount_to_pad = Self::BLOCK_SIZE - (text.len() % Self::BLOCK_SIZE);
        if amount_to_pad == 0 {
            amount_to_pad = Self::BLOCK_SIZE;
        }
        let mut out = Vec::with_capacity(text.len() + amount_to_pad);
        out.extend_from_slice(text);
        out.extend(std::iter::repeat(amount_to_pad as u8).take(amount_to_pad));
        out
    }

    /// Strip PKCS#7 padding from `decrypted`.
    pub fn decode(decrypted: &[u8]) -> Result<Vec<u8>> {
        if decrypted.is_empty() {
            return Err(WeComCryptoError::Decrypt("empty decrypted payload".into()));
        }
        let pad = *decrypted.last().unwrap() as usize;
        if pad < 1 || pad > Self::BLOCK_SIZE {
            return Err(WeComCryptoError::Decrypt("invalid PKCS7 padding".into()));
        }
        if pad > decrypted.len() {
            return Err(WeComCryptoError::Decrypt("invalid PKCS7 padding".into()));
        }
        let tail = &decrypted[decrypted.len() - pad..];
        if tail.iter().any(|&b| b as usize != pad) {
            return Err(WeComCryptoError::Decrypt("malformed PKCS7 padding".into()));
        }
        Ok(decrypted[..decrypted.len() - pad].to_vec())
    }
}

/// Compute the WeCom SHA-1 signature over the sorted concatenation of
/// `[token, timestamp, nonce, encrypt]`, returned as a lowercase hex digest.
pub fn sha1_signature(token: &str, timestamp: &str, nonce: &str, encrypt: &str) -> String {
    let mut parts = [token, timestamp, nonce, encrypt];
    parts.sort_unstable();
    let mut hasher = Sha1::new();
    for p in parts {
        hasher.update(p.as_bytes());
    }
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Minimal WeCom callback crypto helper compatible with `BizMsgCrypt` semantics.
pub struct WXBizMsgCrypt {
    token: String,
    receive_id: String,
    key: [u8; 32],
    iv: [u8; 16],
}

impl WXBizMsgCrypt {
    /// Construct a new helper.
    ///
    /// `encoding_aes_key` must be exactly 43 characters; it is base64-decoded
    /// (after appending `"="`) into the 32-byte AES-256 key.
    pub fn new(token: &str, encoding_aes_key: &str, receive_id: &str) -> Result<Self> {
        if token.is_empty() {
            return Err(WeComCryptoError::Value("token is required".into()));
        }
        if encoding_aes_key.is_empty() {
            return Err(WeComCryptoError::Value("encoding_aes_key is required".into()));
        }
        if encoding_aes_key.chars().count() != 43 {
            return Err(WeComCryptoError::Value(
                "encoding_aes_key must be 43 chars".into(),
            ));
        }
        if receive_id.is_empty() {
            return Err(WeComCryptoError::Value("receive_id is required".into()));
        }

        let decoded = base64::engine::general_purpose::STANDARD
            .decode(format!("{encoding_aes_key}="))
            .map_err(|e| WeComCryptoError::Value(format!("invalid encoding_aes_key: {e}")))?;
        if decoded.len() != 32 {
            return Err(WeComCryptoError::Value(
                "encoding_aes_key must decode to 32 bytes".into(),
            ));
        }
        let mut key = [0u8; 32];
        key.copy_from_slice(&decoded);
        let mut iv = [0u8; 16];
        iv.copy_from_slice(&decoded[..16]);

        Ok(Self {
            token: token.to_string(),
            receive_id: receive_id.to_string(),
            key,
            iv,
        })
    }

    /// Verify a callback `verify_url` request and return the decrypted echo
    /// string as UTF-8.
    pub fn verify_url(
        &self,
        msg_signature: &str,
        timestamp: &str,
        nonce: &str,
        echostr: &str,
    ) -> Result<String> {
        let plain = self.decrypt(msg_signature, timestamp, nonce, echostr)?;
        String::from_utf8(plain)
            .map_err(|e| WeComCryptoError::Decrypt(format!("invalid utf-8: {e}")))
    }

    /// Verify the signature and decrypt the `encrypt` field, returning the raw
    /// XML content bytes.
    pub fn decrypt(
        &self,
        msg_signature: &str,
        timestamp: &str,
        nonce: &str,
        encrypt: &str,
    ) -> Result<Vec<u8>> {
        let expected = sha1_signature(&self.token, timestamp, nonce, encrypt);
        if expected != msg_signature {
            return Err(WeComCryptoError::Signature("signature mismatch".into()));
        }

        let cipher_text = base64::engine::general_purpose::STANDARD
            .decode(encrypt)
            .map_err(|e| WeComCryptoError::Decrypt(format!("invalid base64 payload: {e}")))?;

        let padded = self.aes_cbc_decrypt(&cipher_text)?;
        let plain = Pkcs7Encoder::decode(&padded)?;

        // skip 16-byte random prefix
        if plain.len() < 16 {
            return Err(WeComCryptoError::Decrypt(
                "decrypt failed: payload too short".into(),
            ));
        }
        let content = &plain[16..];
        if content.len() < 4 {
            return Err(WeComCryptoError::Decrypt(
                "decrypt failed: missing length header".into(),
            ));
        }
        // network byte order (big-endian) 32-bit length
        let xml_length = u32::from_be_bytes([content[0], content[1], content[2], content[3]]) as usize;
        let body = &content[4..];
        if xml_length > body.len() {
            return Err(WeComCryptoError::Decrypt(
                "decrypt failed: declared length exceeds payload".into(),
            ));
        }
        let xml_content = body[..xml_length].to_vec();
        let receive_id = String::from_utf8(body[xml_length..].to_vec())
            .map_err(|e| WeComCryptoError::Decrypt(format!("decrypt failed: {e}")))?;

        if receive_id != self.receive_id {
            return Err(WeComCryptoError::Decrypt("receive_id mismatch".into()));
        }
        Ok(xml_content)
    }

    /// Encrypt `plaintext` and return a WeCom-formatted response XML document
    /// containing `Encrypt`, `MsgSignature`, `TimeStamp`, and `Nonce` elements.
    ///
    /// `nonce` / `timestamp` default to a fresh random nonce and the current
    /// UNIX time when `None`.
    pub fn encrypt(
        &self,
        plaintext: &str,
        nonce: Option<&str>,
        timestamp: Option<&str>,
    ) -> Result<String> {
        let nonce_owned;
        let nonce = match nonce {
            Some(n) => n.to_string(),
            None => {
                nonce_owned = Self::random_nonce(10);
                nonce_owned
            }
        };
        let timestamp = match timestamp {
            Some(t) => t.to_string(),
            None => current_unix_secs().to_string(),
        };

        let encrypt = self.encrypt_bytes(plaintext.as_bytes())?;
        let signature = sha1_signature(&self.token, &timestamp, &nonce, &encrypt);

        Ok(format!(
            "<xml><Encrypt>{enc}</Encrypt><MsgSignature>{sig}</MsgSignature><TimeStamp>{ts}</TimeStamp><Nonce>{nonce}</Nonce></xml>",
            enc = xml_escape(&encrypt),
            sig = xml_escape(&signature),
            ts = xml_escape(&timestamp),
            nonce = xml_escape(&nonce),
        ))
    }

    /// Build the AES-encrypted, base64-encoded `Encrypt` field for `raw`.
    fn encrypt_bytes(&self, raw: &[u8]) -> Result<String> {
        let random_prefix = random_bytes(16);
        let msg_len = (raw.len() as u32).to_be_bytes();

        let mut payload =
            Vec::with_capacity(16 + 4 + raw.len() + self.receive_id.len());
        payload.extend_from_slice(&random_prefix);
        payload.extend_from_slice(&msg_len);
        payload.extend_from_slice(raw);
        payload.extend_from_slice(self.receive_id.as_bytes());

        let padded = Pkcs7Encoder::encode(&payload);
        let encrypted = self.aes_cbc_encrypt(&padded)?;
        Ok(base64::engine::general_purpose::STANDARD.encode(encrypted))
    }

    fn aes_cbc_encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.len() % BLOCK_SIZE != 0 {
            return Err(WeComCryptoError::Encrypt(
                "encrypt failed: input not block-aligned".into(),
            ));
        }
        let cipher = Aes256::new(&Array(self.key));
        let mut out = Vec::with_capacity(data.len());
        let mut prev = self.iv;
        for chunk in data.chunks(BLOCK_SIZE) {
            let mut block = [0u8; BLOCK_SIZE];
            for i in 0..BLOCK_SIZE {
                block[i] = chunk[i] ^ prev[i];
            }
            let mut arr = Array(block);
            cipher.encrypt_block(&mut arr);
            prev = arr.0;
            out.extend_from_slice(&arr.0);
        }
        Ok(out)
    }

    fn aes_cbc_decrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() || data.len() % BLOCK_SIZE != 0 {
            return Err(WeComCryptoError::Decrypt(
                "decrypt failed: cipher text not block-aligned".into(),
            ));
        }
        let cipher = Aes256::new(&Array(self.key));
        let mut out = Vec::with_capacity(data.len());
        let mut prev = self.iv;
        for chunk in data.chunks(BLOCK_SIZE) {
            let mut ciph = [0u8; BLOCK_SIZE];
            ciph.copy_from_slice(chunk);
            let mut arr = Array(ciph);
            cipher.decrypt_block(&mut arr);
            for i in 0..BLOCK_SIZE {
                out.push(arr.0[i] ^ prev[i]);
            }
            prev = ciph;
        }
        Ok(out)
    }

    /// Generate a random alphanumeric nonce of `length` characters.
    pub fn random_nonce(length: usize) -> String {
        const ALPHABET: &[u8] =
            b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let raw = random_bytes(length);
        raw.iter()
            .map(|&b| ALPHABET[(b as usize) % ALPHABET.len()] as char)
            .collect()
    }
}

/// XML-escape a text node value.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(c),
        }
    }
    out
}

/// Current UNIX time in whole seconds.
fn current_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Return `n` cryptographically-sourced random bytes.
///
/// Reads from the OS CSPRNG (`/dev/urandom` on unix). Falls back to a
/// time/address-seeded xorshift generator only if the OS source is
/// unavailable, which keeps the helper usable in restricted sandboxes.
fn random_bytes(n: usize) -> Vec<u8> {
    if let Some(bytes) = os_random_bytes(n) {
        return bytes;
    }
    fallback_random_bytes(n)
}

#[cfg(unix)]
fn os_random_bytes(n: usize) -> Option<Vec<u8>> {
    use std::io::Read;
    let mut f = std::fs::File::open("/dev/urandom").ok()?;
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf).ok()?;
    Some(buf)
}

#[cfg(not(unix))]
fn os_random_bytes(_n: usize) -> Option<Vec<u8>> {
    None
}

fn fallback_random_bytes(n: usize) -> Vec<u8> {
    let mut state = {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        let addr = &state_seed as *const u8 as u64;
        nanos ^ addr.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0xD1B5_4A32_D192_ED03
    };
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push((state & 0xff) as u8);
    }
    out
}

/// Anchor used to derive an address-based entropy seed for the fallback RNG.
static state_seed: u8 = 0;

#[cfg(test)]
mod tests {
    use super::*;

    // Test key: 43 base64 chars -> 32 bytes after appending '='.
    // "0123456789012345678901234567890123456789012" is 43 chars.
    const TEST_KEY: &str = "0123456789012345678901234567890123456789012";
    const TOKEN: &str = "QDG6eK";
    const RECEIVE_ID: &str = "wx5823bf96d3bd56c7";

    fn helper() -> WXBizMsgCrypt {
        WXBizMsgCrypt::new(TOKEN, TEST_KEY, RECEIVE_ID).unwrap()
    }

    #[test]
    fn rejects_bad_constructor_args() {
        assert!(matches!(
            WXBizMsgCrypt::new("", TEST_KEY, RECEIVE_ID),
            Err(WeComCryptoError::Value(_))
        ));
        assert!(matches!(
            WXBizMsgCrypt::new(TOKEN, "tooshort", RECEIVE_ID),
            Err(WeComCryptoError::Value(_))
        ));
        assert!(matches!(
            WXBizMsgCrypt::new(TOKEN, TEST_KEY, ""),
            Err(WeComCryptoError::Value(_))
        ));
        // exactly 43 chars required
        let short42 = &TEST_KEY[..42];
        assert!(matches!(
            WXBizMsgCrypt::new(TOKEN, short42, RECEIVE_ID),
            Err(WeComCryptoError::Value(_))
        ));
    }

    #[test]
    fn key_and_iv_derivation() {
        let h = helper();
        assert_eq!(h.key.len(), 32);
        assert_eq!(&h.iv[..], &h.key[..16]);
    }

    #[test]
    fn pkcs7_roundtrip_block_aligned() {
        // 32-byte input must gain a full extra 32-byte pad block (amount_to_pad == 0 -> BLOCK_SIZE)
        let data = vec![1u8; 32];
        let enc = Pkcs7Encoder::encode(&data);
        assert_eq!(enc.len(), 64);
        assert_eq!(*enc.last().unwrap(), 32);
        let dec = Pkcs7Encoder::decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn pkcs7_roundtrip_partial() {
        let data = b"hello".to_vec();
        let enc = Pkcs7Encoder::encode(&data);
        assert_eq!(enc.len(), 32);
        assert_eq!(*enc.last().unwrap(), 27);
        let dec = Pkcs7Encoder::decode(&enc).unwrap();
        assert_eq!(dec, data);
    }

    #[test]
    fn pkcs7_rejects_bad_padding() {
        assert!(Pkcs7Encoder::decode(&[]).is_err());
        // last byte 0 -> invalid
        assert!(Pkcs7Encoder::decode(&[1, 2, 0]).is_err());
        // last byte > block size
        assert!(Pkcs7Encoder::decode(&[33]).is_err());
        // declared pad larger than buffer
        assert!(Pkcs7Encoder::decode(&[5]).is_err());
        // malformed: claims 3 bytes of pad but they aren't all 3
        assert!(Pkcs7Encoder::decode(&[3, 3, 2]).is_err());
    }

    #[test]
    fn signature_is_sorted_sha1() {
        // Known sort + concat behavior.
        let sig = sha1_signature("b", "a", "d", "c");
        // sorted -> a b c d -> "abcd"
        let mut hasher = Sha1::new();
        hasher.update(b"abcd");
        let expected: String =
            hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(sig, expected);
        assert_eq!(sig.len(), 40);
    }

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let h = helper();
        let plaintext = "<xml><Content>hello wecom</Content></xml>";
        let nonce = "1372623149";
        let timestamp = "1409304348";

        let xml = h.encrypt(plaintext, Some(nonce), Some(timestamp)).unwrap();
        assert!(xml.contains("<Encrypt>"));
        assert!(xml.contains("<MsgSignature>"));
        assert!(xml.contains(&format!("<TimeStamp>{timestamp}</TimeStamp>")));
        assert!(xml.contains(&format!("<Nonce>{nonce}</Nonce>")));

        // Extract the Encrypt and MsgSignature values back out.
        let encrypt = extract(&xml, "Encrypt");
        let signature = extract(&xml, "MsgSignature");

        // Signature must match recomputation.
        assert_eq!(
            signature,
            sha1_signature(TOKEN, timestamp, nonce, &encrypt)
        );

        // And the helper must decrypt it back to the original plaintext.
        let recovered = h
            .decrypt(&signature, timestamp, nonce, &encrypt)
            .unwrap();
        assert_eq!(recovered, plaintext.as_bytes());

        let echo = h.verify_url(&signature, timestamp, nonce, &encrypt).unwrap();
        assert_eq!(echo, plaintext);
    }

    #[test]
    fn decrypt_rejects_bad_signature() {
        let h = helper();
        let xml = h
            .encrypt("data", Some("nonce123"), Some("100"))
            .unwrap();
        let encrypt = extract(&xml, "Encrypt");
        let err = h.decrypt("deadbeef", "100", "nonce123", &encrypt).unwrap_err();
        assert!(matches!(err, WeComCryptoError::Signature(_)));
    }

    #[test]
    fn decrypt_rejects_receive_id_mismatch() {
        let producer = helper();
        let xml = producer
            .encrypt("payload", Some("nn"), Some("5"))
            .unwrap();
        let encrypt = extract(&xml, "Encrypt");

        // Different receive_id consumer: signature still recomputable so we
        // must reach the receive_id check.
        let other = WXBizMsgCrypt::new(TOKEN, TEST_KEY, "different_corp").unwrap();
        let sig = sha1_signature(TOKEN, "5", "nn", &encrypt);
        let err = other.decrypt(&sig, "5", "nn", &encrypt).unwrap_err();
        assert!(matches!(err, WeComCryptoError::Decrypt(_)));
        assert_eq!(err.to_string(), "receive_id mismatch");
    }

    #[test]
    fn random_nonce_length_and_alphabet() {
        let n = WXBizMsgCrypt::random_nonce(10);
        assert_eq!(n.chars().count(), 10);
        assert!(n
            .chars()
            .all(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn xml_escaping() {
        assert_eq!(xml_escape("a&b<c>d"), "a&amp;b&lt;c&gt;d");
    }

    fn extract(xml: &str, tag: &str) -> String {
        let open = format!("<{tag}>");
        let close = format!("</{tag}>");
        let start = xml.find(&open).unwrap() + open.len();
        let end = xml[start..].find(&close).unwrap() + start;
        xml[start..end].to_string()
    }
}
