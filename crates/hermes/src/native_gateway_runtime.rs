use std::error::Error;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener};
use std::thread;
use std::time::Duration;

use hermes_core::{HermesContext, LoadedConfig};
use serde_json::{Value, json};
use serde_yaml::Value as YamlValue;
use tokio::runtime::Runtime;
use tokio::sync::watch;
use tokio::task::JoinSet;

use crate::gateway_cmd::GatewayRunArgs;
use crate::native_api_server::{
    NativeApiServerState, load_native_api_server_state, serve_native_api_server,
};
use crate::native_webhook_server::{
    NativeWebhookState, load_native_webhook_state, serve_native_webhook_server,
};

pub(crate) fn maybe_run_native_gateway_bundle(
    context: &HermesContext,
    args: &GatewayRunArgs,
) -> Result<bool, Box<dyn Error>> {
    let loaded = context.load_config_document()?;
    let api_state = load_native_api_server_state(context, &loaded)?;
    let webhook_state = load_native_webhook_state(context, &loaded)?;
    if api_state.is_none() && webhook_state.is_none() {
        return Ok(false);
    }
    if has_non_native_bundle_platforms_enabled(&loaded) {
        return Ok(false);
    }
    run_native_gateway_bundle(api_state, webhook_state, args)?;
    Ok(true)
}

fn run_native_gateway_bundle(
    api_state: Option<NativeApiServerState>,
    webhook_state: Option<NativeWebhookState>,
    _args: &GatewayRunArgs,
) -> Result<(), Box<dyn Error>> {
    let runtime = Runtime::new()?;
    runtime.block_on(async move {
        let mut join_set = JoinSet::new();
        let (shutdown_tx, shutdown_rx) = watch::channel(false);

        if let Some(state) = api_state {
            let bind_ip: IpAddr = state
                .settings
                .host
                .parse()
                .map_err(|_| "API_SERVER_HOST must be a valid IP address for native Rust runtime")?;
            if is_network_accessible(bind_ip) && state.settings.api_key.trim().is_empty() {
                return Err(
                    "Refusing to start native API server on a non-loopback address without API_SERVER_KEY"
                        .into(),
                );
            }
            let listener =
                tokio::net::TcpListener::bind(SocketAddr::new(bind_ip, state.settings.port)).await?;
            println!(
                "Native API server listening on http://{}:{} (model: {})",
                state.settings.host, state.settings.port, state.settings.model_name
            );
            let mut shutdown = shutdown_rx.clone();
            join_set.spawn(async move {
                serve_native_api_server(listener, state, async move {
                    let _ = shutdown.changed().await;
                })
                .await
                .map_err(|error| error.to_string())
            });
        }

        if let Some(state) = webhook_state {
            let bind_ip: IpAddr = state
                .settings
                .host
                .parse()
                .map_err(|_| "WEBHOOK_HOST must be a valid IP address for native Rust runtime")?;
            let listener =
                tokio::net::TcpListener::bind(SocketAddr::new(bind_ip, state.settings.port)).await?;
            println!(
                "Native webhook server listening on http://{}:{}",
                state.settings.host, state.settings.port
            );
            let mut shutdown = shutdown_rx.clone();
            join_set.spawn(async move {
                serve_native_webhook_server(listener, state, async move {
                    let _ = shutdown.changed().await;
                })
                .await
                .map_err(|error| error.to_string())
            });
        }

        if join_set.is_empty() {
            return Ok(());
        }

        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                let _ = shutdown_tx.send(true);
            }
            result = join_set.join_next() => {
                let _ = shutdown_tx.send(true);
                match result {
                    Some(Ok(Ok(()))) => {
                        return Err(io::Error::other("native gateway server exited unexpectedly").into());
                    }
                    Some(Ok(Err(error))) => {
                        return Err(io::Error::other(error).into());
                    }
                    Some(Err(error)) => {
                        return Err(io::Error::other(error.to_string()).into());
                    }
                    None => {}
                }
            }
        }

        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(Ok(())) => {}
                Ok(Err(error)) => return Err(io::Error::other(error).into()),
                Err(error) => return Err(io::Error::other(error.to_string()).into()),
            }
        }

        Ok(())
    })
}

fn has_non_native_bundle_platforms_enabled(loaded: &LoadedConfig) -> bool {
    let env_enabled = [
        ("TELEGRAM_BOT_TOKEN", false),
        ("DISCORD_BOT_TOKEN", false),
        ("SLACK_BOT_TOKEN", false),
        ("FEISHU_APP_ID", false),
        ("MATRIX_ACCESS_TOKEN", false),
        ("WHATSAPP_ENABLED", true),
        ("DINGTALK_CLIENT_ID", false),
        ("QQ_APP_ID", false),
        ("MATTERMOST_TOKEN", false),
        ("WECOM_BOT_ID", false),
        ("WEIXIN_TOKEN", false),
        ("EMAIL_ADDRESS", false),
        ("TWILIO_ACCOUNT_SID", false),
        ("HASS_TOKEN", false),
        ("BLUEBUBBLES_SERVER_URL", false),
        ("SIGNAL_HTTP_URL", false),
        ("YUANBAO_APP_ID", false),
    ]
    .iter()
    .any(|(key, truthy)| {
        if *truthy {
            env_truthy(key)
        } else {
            env_string(key).is_some()
        }
    });
    if env_enabled {
        return true;
    }
    loaded
        .cfg_get(&["platforms"])
        .and_then(YamlValue::as_mapping)
        .is_some_and(|platforms| {
            platforms.iter().any(|(key, value)| {
                let Some(name) = key.as_str() else {
                    return false;
                };
                if matches!(name, "api_server" | "webhook") {
                    return false;
                }
                value
                    .as_mapping()
                    .and_then(|mapping| mapping.get(YamlValue::String("enabled".to_string())))
                    .and_then(yaml_bool)
                    .unwrap_or(false)
            })
        })
}

fn env_truthy(key: &str) -> bool {
    env_string(key).is_some_and(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn env_string(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn yaml_bool(value: &YamlValue) -> Option<bool> {
    match value {
        YamlValue::Bool(boolean) => Some(*boolean),
        YamlValue::String(text) => Some(matches!(
            text.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )),
        _ => None,
    }
}

fn is_network_accessible(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(value) => value != Ipv4Addr::LOCALHOST,
        IpAddr::V6(value) => value != Ipv6Addr::LOCALHOST,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    use std::fs;
    use std::io::{Read, Write};
    use tempfile::TempDir;
    use tokio::sync::watch;

    type HmacSha256 = Hmac<Sha256>;

    fn temp_context(config_text: &str) -> (TempDir, HermesContext, LoadedConfig) {
        let temp = TempDir::new().unwrap();
        fs::write(temp.path().join("config.yaml"), config_text).unwrap();
        let context =
            HermesContext::new(temp.path()).with_hermes_home_env(Some(temp.path().to_path_buf()));
        let loaded = context.load_config_document().unwrap();
        (temp, context, loaded)
    }

    fn mock_http_server<F>(handler: F) -> (String, thread::JoinHandle<()>)
    where
        F: FnOnce(String, String) + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let header_end = request
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|value| value + 4)
                .unwrap();
            let headers = String::from_utf8_lossy(&request[..header_end]).to_string();
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    if name.eq_ignore_ascii_case("content-length") {
                        value.trim().parse::<usize>().ok()
                    } else {
                        None
                    }
                })
                .unwrap_or(0);
            let mut body = request[header_end..].to_vec();
            while body.len() < content_length {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                body.extend_from_slice(&buffer[..read]);
            }
            handler(
                headers,
                String::from_utf8_lossy(&body[..content_length]).to_string(),
            );
            let response_body = "{\"errcode\":0,\"errmsg\":\"ok\"}";
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        });
        (format!("http://{}", addr), join)
    }

    fn mock_model_server(response_body: String) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let join = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        });
        (format!("http://{}", addr), join)
    }

    fn compute_signature(secret: &str, body: &[u8]) -> String {
        let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(body);
        let bytes = mac.finalize().into_bytes();
        format!(
            "sha256={}",
            bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        )
    }

    #[test]
    fn combined_native_gateway_serves_api_server_and_webhook() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let (dingtalk_base_url, dingtalk_join) = mock_http_server(|headers, body| {
            assert!(headers.starts_with("POST /robot/send?access_token=test-token "));
            let payload: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(payload["text"]["content"], json!("Alert: ping"));
        });
        let (model_base_url, model_join) = mock_model_server(
            json!({
                "id": "chatcmpl-test",
                "choices": [{
                    "message": {
                        "content": "bundle hello"
                    }
                }]
            })
            .to_string(),
        );
        let webhook_url = format!("{dingtalk_base_url}/robot/send?access_token=test-token");
        unsafe {
            std::env::set_var("DINGTALK_WEBHOOK_URL", &webhook_url);
        }

        let config_text = format!(
            "model:\n  default: test-model\n  provider: custom\n  base_url: {model_base_url}\n  api_key: test-key\n  api_mode: chat_completions\nplatforms:\n  api_server:\n    enabled: true\n    extra:\n      host: 127.0.0.1\n      port: 8642\n  webhook:\n    enabled: true\n    extra:\n      host: 127.0.0.1\n      port: 8644\n      routes:\n        alerts:\n          secret: topsecret\n          prompt: 'Alert: {{message}}'\n          deliver: dingtalk\n          deliver_only: true\n          deliver_extra:\n            chat_id: cidding==\n"
        );
        let (_temp, context, loaded) = temp_context(&config_text);
        let api_state = load_native_api_server_state(&context, &loaded)
            .unwrap()
            .unwrap();
        let webhook_state = load_native_webhook_state(&context, &loaded)
            .unwrap()
            .unwrap();

        let api_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let api_addr = api_listener.local_addr().unwrap();
        drop(api_listener);
        let webhook_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let webhook_addr = webhook_listener.local_addr().unwrap();
        drop(webhook_listener);

        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let server_thread = thread::spawn(move || {
            let runtime = Runtime::new().unwrap();
            runtime.block_on(async move {
                let api_listener = tokio::net::TcpListener::bind(api_addr).await.unwrap();
                let webhook_listener = tokio::net::TcpListener::bind(webhook_addr).await.unwrap();
                let mut servers = JoinSet::new();
                let mut root_shutdown = shutdown_rx.clone();

                let mut api_shutdown = shutdown_rx.clone();
                servers.spawn(async move {
                    serve_native_api_server(api_listener, api_state, async move {
                        let _ = api_shutdown.changed().await;
                    })
                    .await
                    .map_err(|error| error.to_string())
                });

                let mut webhook_shutdown = shutdown_rx.clone();
                servers.spawn(async move {
                    serve_native_webhook_server(webhook_listener, webhook_state, async move {
                        let _ = webhook_shutdown.changed().await;
                    })
                    .await
                    .map_err(|error| error.to_string())
                });

                tokio::select! {
                    result = servers.join_next() => {
                        match result {
                            Some(Ok(Ok(()))) => panic!("server exited unexpectedly"),
                            Some(Ok(Err(error))) => panic!("{error}"),
                            Some(Err(error)) => panic!("{error}"),
                            None => {}
                        }
                    }
                    _ = root_shutdown.changed() => {}
                }

                while let Some(result) = servers.join_next().await {
                    match result {
                        Ok(Ok(())) => {}
                        Ok(Err(error)) => panic!("{error}"),
                        Err(error) => panic!("{error}"),
                    }
                }
            });
        });

        let client = reqwest::blocking::Client::new();
        for _ in 0..20 {
            if client
                .get(format!("http://{api_addr}/health"))
                .send()
                .is_ok()
                && client
                    .get(format!("http://{webhook_addr}/health"))
                    .send()
                    .is_ok()
            {
                break;
            }
            thread::sleep(Duration::from_millis(25));
        }

        let api_response: Value = client
            .post(format!("http://{api_addr}/v1/chat/completions"))
            .json(&json!({
                "messages": [{
                    "role": "user",
                    "content": "say hi"
                }]
            }))
            .send()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(
            api_response["choices"][0]["message"]["content"],
            json!("bundle hello")
        );

        let payload = br#"{"message":"ping"}"#;
        let webhook_response: Value = client
            .post(format!("http://{webhook_addr}/webhooks/alerts"))
            .header("Content-Type", "application/json")
            .header(
                "X-Hub-Signature-256",
                compute_signature("topsecret", payload),
            )
            .header("X-GitHub-Event", "test")
            .body(payload.to_vec())
            .send()
            .unwrap()
            .json()
            .unwrap();
        assert_eq!(webhook_response["status"], json!("delivered"));

        drop(client);
        let _ = shutdown_tx.send(true);
        server_thread.join().unwrap();
        dingtalk_join.join().unwrap();
        model_join.join().unwrap();
        unsafe {
            std::env::remove_var("DINGTALK_WEBHOOK_URL");
        }
    }

    #[test]
    fn combined_native_gateway_rejects_other_enabled_platforms() {
        let _guard = crate::cli_test_env_lock().lock().unwrap();
        let (_temp, _context, loaded) = temp_context(
            "platforms:\n  api_server:\n    enabled: true\n  webhook:\n    enabled: true\n  telegram:\n    enabled: true\n",
        );
        assert!(has_non_native_bundle_platforms_enabled(&loaded));
    }
}
