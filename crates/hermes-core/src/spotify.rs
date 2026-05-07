use std::env;
use std::fmt::{self, Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, Utc};
use reqwest::Method;
use reqwest::blocking::Client;
use serde_json::{Map, Value, json};
use url::Url;

use crate::tools::{ToolRuntime, tool_error, tool_result};

const DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL: &str = "https://accounts.spotify.com";
const DEFAULT_SPOTIFY_API_BASE_URL: &str = "https://api.spotify.com/v1";
const DEFAULT_SPOTIFY_REDIRECT_URI: &str = "http://127.0.0.1:43827/spotify/callback";
const DEFAULT_SPOTIFY_SCOPE: &str = "user-modify-playback-state user-read-playback-state user-read-currently-playing user-read-recently-played playlist-read-private playlist-read-collaborative playlist-modify-public playlist-modify-private user-library-read user-library-modify";
const SPOTIFY_ACCESS_TOKEN_REFRESH_SKEW_SECONDS: i64 = 120;
const SPOTIFY_REQUEST_TIMEOUT_SECS: u64 = 30;
const SPOTIFY_REFRESH_TIMEOUT_SECS: u64 = 20;

#[derive(Debug, Clone)]
struct SpotifyRuntimeCredentials {
    access_token: String,
    token_type: String,
    base_url: String,
}

#[derive(Debug, Clone)]
struct SpotifyClient {
    home: PathBuf,
    runtime: SpotifyRuntimeCredentials,
}

#[derive(Debug, Clone)]
struct SpotifyApiError {
    message: String,
    status_code: Option<u16>,
    response_body: Option<String>,
}

impl Display for SpotifyApiError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

#[derive(Debug, Clone)]
enum SpotifyError {
    Message(String),
    Api(SpotifyApiError),
}

impl Display for SpotifyError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Message(message) => f.write_str(message),
            Self::Api(error) => Display::fmt(error, f),
        }
    }
}

pub fn spotify_available() -> bool {
    spotify_auth_logged_in(&hermes_home_from_env())
}

pub fn spotify_playback_schema() -> Value {
    json!({
        "name": "spotify_playback",
        "description": "Control Spotify playback, inspect the active playback state, or fetch recently played tracks.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["get_state", "get_currently_playing", "play", "pause", "next", "previous", "seek", "set_repeat", "set_shuffle", "set_volume", "recently_played"]},
                "device_id": {"type": "string"},
                "market": {"type": "string"},
                "context_uri": {"type": "string"},
                "uris": {"type": "array", "items": {"type": "string"}},
                "offset": {"type": "object"},
                "position_ms": {"type": "integer"},
                "state": {"description": "For set_repeat use track/context/off. For set_shuffle use boolean-like true/false.", "oneOf": [{"type": "string"}, {"type": "boolean"}]},
                "volume_percent": {"type": "integer"},
                "limit": {"type": "integer", "description": "For recently_played: number of tracks (max 50)"},
                "after": {"type": "integer", "description": "For recently_played: Unix ms cursor (after this timestamp)"},
                "before": {"type": "integer", "description": "For recently_played: Unix ms cursor (before this timestamp)"}
            },
            "required": ["action"]
        }
    })
}

pub fn spotify_devices_schema() -> Value {
    json!({
        "name": "spotify_devices",
        "description": "List Spotify Connect devices or transfer playback to a different device.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "transfer"]},
                "device_id": {"type": "string"},
                "play": {"type": "boolean"}
            },
            "required": ["action"]
        }
    })
}

pub fn spotify_queue_schema() -> Value {
    json!({
        "name": "spotify_queue",
        "description": "Inspect the user's Spotify queue or add an item to it.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["get", "add"]},
                "uri": {"type": "string"},
                "device_id": {"type": "string"}
            },
            "required": ["action"]
        }
    })
}

pub fn spotify_search_schema() -> Value {
    json!({
        "name": "spotify_search",
        "description": "Search the Spotify catalog for tracks, albums, artists, playlists, shows, or episodes.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "types": {"type": "array", "items": {"type": "string"}},
                "type": {"type": "string"},
                "limit": {"type": "integer"},
                "offset": {"type": "integer"},
                "market": {"type": "string"},
                "include_external": {"type": "string"}
            },
            "required": ["query"]
        }
    })
}

pub fn spotify_playlists_schema() -> Value {
    json!({
        "name": "spotify_playlists",
        "description": "List, inspect, create, update, and modify Spotify playlists.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "get", "create", "add_items", "remove_items", "update_details"]},
                "playlist_id": {"type": "string"},
                "market": {"type": "string"},
                "limit": {"type": "integer"},
                "offset": {"type": "integer"},
                "name": {"type": "string"},
                "description": {"type": "string"},
                "public": {"type": "boolean"},
                "collaborative": {"type": "boolean"},
                "uris": {"type": "array", "items": {"type": "string"}},
                "position": {"type": "integer"},
                "snapshot_id": {"type": "string"}
            },
            "required": ["action"]
        }
    })
}

pub fn spotify_albums_schema() -> Value {
    json!({
        "name": "spotify_albums",
        "description": "Fetch Spotify album metadata or album tracks.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["get", "tracks"]},
                "album_id": {"type": "string"},
                "id": {"type": "string"},
                "market": {"type": "string"},
                "limit": {"type": "integer"},
                "offset": {"type": "integer"}
            },
            "required": ["action"]
        }
    })
}

pub fn spotify_library_schema() -> Value {
    json!({
        "name": "spotify_library",
        "description": "List, save, or remove the user's saved Spotify tracks or albums. Use kind to select which.",
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["tracks", "albums"], "description": "Which library to operate on"},
                "action": {"type": "string", "enum": ["list", "save", "remove"]},
                "limit": {"type": "integer"},
                "offset": {"type": "integer"},
                "market": {"type": "string"},
                "uris": {"type": "array", "items": {"type": "string"}},
                "ids": {"type": "array", "items": {"type": "string"}},
                "items": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["kind", "action"]
        }
    })
}

pub fn handle_spotify_playback(args: &Value, runtime: &ToolRuntime) -> String {
    let action = lower_string(args.get("action")).unwrap_or_else(|| "get_state".to_string());
    let client = match spotify_client(runtime.hermes_home()) {
        Ok(client) => client,
        Err(error) => return spotify_tool_error(error),
    };

    let result = match action.as_str() {
        "get_state" => client.get_playback_state(optional_string(args.get("market"))),
        "get_currently_playing" => {
            client.get_currently_playing(optional_string(args.get("market")))
        }
        "play" => {
            let offset = args.get("offset").cloned().filter(Value::is_object);
            let uris = if args.get("uris").is_some() {
                Some(
                    match normalize_spotify_uris(as_list(args.get("uris")), Some("track")) {
                        Ok(value) => value,
                        Err(error) => return spotify_tool_error(SpotifyError::Message(error)),
                    },
                )
            } else {
                None
            };
            let context_uri = match optional_string(args.get("context_uri")) {
                Some(raw_context) => {
                    let context_type = if raw_context.starts_with("spotify:album:")
                        || raw_context.contains("/album/")
                    {
                        Some("album")
                    } else if raw_context.starts_with("spotify:playlist:")
                        || raw_context.contains("/playlist/")
                    {
                        Some("playlist")
                    } else if raw_context.starts_with("spotify:artist:")
                        || raw_context.contains("/artist/")
                    {
                        Some("artist")
                    } else {
                        None
                    };
                    Some(match normalize_spotify_uri(&raw_context, context_type) {
                        Ok(value) => value,
                        Err(error) => return spotify_tool_error(SpotifyError::Message(error)),
                    })
                }
                None => None,
            };
            client
                .start_playback(
                    optional_string(args.get("device_id")),
                    context_uri,
                    uris,
                    offset,
                    optional_i64(args.get("position_ms")),
                )
                .map(|value| json!({"success": true, "action": action, "result": value}))
        }
        "pause" => client
            .pause_playback(optional_string(args.get("device_id")))
            .map(|value| json!({"success": true, "action": action, "result": value})),
        "next" => client
            .skip_next(optional_string(args.get("device_id")))
            .map(|value| json!({"success": true, "action": action, "result": value})),
        "previous" => client
            .skip_previous(optional_string(args.get("device_id")))
            .map(|value| json!({"success": true, "action": action, "result": value})),
        "seek" => {
            let Some(position_ms) = optional_i64(args.get("position_ms")) else {
                return tool_error("position_ms is required for action='seek'");
            };
            client
                .seek(position_ms, optional_string(args.get("device_id")))
                .map(|value| json!({"success": true, "action": action, "result": value}))
        }
        "set_repeat" => {
            let state = lower_string(args.get("state")).unwrap_or_default();
            if !matches!(state.as_str(), "track" | "context" | "off") {
                return tool_error("state must be one of: track, context, off");
            }
            client
                .set_repeat(&state, optional_string(args.get("device_id")))
                .map(|value| json!({"success": true, "action": action, "result": value}))
        }
        "set_shuffle" => client
            .set_shuffle(
                coerce_bool(args.get("state"), false),
                optional_string(args.get("device_id")),
            )
            .map(|value| json!({"success": true, "action": action, "result": value})),
        "set_volume" => {
            let Some(volume_percent) = optional_i64(args.get("volume_percent")) else {
                return tool_error("volume_percent is required for action='set_volume'");
            };
            client
                .set_volume(
                    volume_percent.clamp(0, 100),
                    optional_string(args.get("device_id")),
                )
                .map(|value| json!({"success": true, "action": action, "result": value}))
        }
        "recently_played" => {
            let after = optional_i64(args.get("after"));
            let before = optional_i64(args.get("before"));
            if after.is_some() && before.is_some() {
                return tool_error("Provide only one of 'after' or 'before'");
            }
            client.get_recently_played(coerce_limit(args.get("limit"), 20, 1, 50), after, before)
        }
        _ => return tool_error(format!("Unknown spotify_playback action: {action}")),
    };

    match result {
        Ok(payload) => {
            if let Some(empty) = describe_empty_playback(&payload, &action) {
                return tool_result(empty);
            }
            tool_result(payload)
        }
        Err(error) => spotify_tool_error(error),
    }
}

pub fn handle_spotify_devices(args: &Value, runtime: &ToolRuntime) -> String {
    let action = lower_string(args.get("action")).unwrap_or_else(|| "list".to_string());
    let client = match spotify_client(runtime.hermes_home()) {
        Ok(client) => client,
        Err(error) => return spotify_tool_error(error),
    };
    let result = match action.as_str() {
        "list" => client.get_devices(),
        "transfer" => {
            let Some(device_id) = optional_string(args.get("device_id")) else {
                return tool_error("device_id is required for action='transfer'");
            };
            client
                .transfer_playback(&device_id, coerce_bool(args.get("play"), false))
                .map(|value| json!({"success": true, "action": action, "result": value}))
        }
        _ => return tool_error(format!("Unknown spotify_devices action: {action}")),
    };
    match result {
        Ok(payload) => tool_result(payload),
        Err(error) => spotify_tool_error(error),
    }
}

pub fn handle_spotify_queue(args: &Value, runtime: &ToolRuntime) -> String {
    let action = lower_string(args.get("action")).unwrap_or_else(|| "get".to_string());
    let client = match spotify_client(runtime.hermes_home()) {
        Ok(client) => client,
        Err(error) => return spotify_tool_error(error),
    };
    let result = match action.as_str() {
        "get" => client.get_queue(),
        "add" => {
            let Some(uri) = optional_string(args.get("uri")) else {
                return tool_error("uri is required for action='add'");
            };
            let uri = match normalize_spotify_uri(&uri, None) {
                Ok(value) => value,
                Err(error) => return tool_error(error),
            };
            client
                .add_to_queue(&uri, optional_string(args.get("device_id")))
                .map(
                    |value| json!({"success": true, "action": action, "uri": uri, "result": value}),
                )
        }
        _ => return tool_error(format!("Unknown spotify_queue action: {action}")),
    };
    match result {
        Ok(payload) => tool_result(payload),
        Err(error) => spotify_tool_error(error),
    }
}

pub fn handle_spotify_search(args: &Value, runtime: &ToolRuntime) -> String {
    let client = match spotify_client(runtime.hermes_home()) {
        Ok(client) => client,
        Err(error) => return spotify_tool_error(error),
    };
    let Some(query) = optional_string(args.get("query")) else {
        return tool_error("query is required");
    };
    let raw_types = if args.get("types").is_some() {
        as_list(args.get("types"))
    } else if args.get("type").is_some() {
        as_list(args.get("type"))
    } else {
        vec!["track".to_string()]
    };
    let search_types = raw_types
        .into_iter()
        .map(|value| value.to_ascii_lowercase())
        .filter(|value| {
            matches!(
                value.as_str(),
                "album" | "artist" | "playlist" | "track" | "show" | "episode" | "audiobook"
            )
        })
        .collect::<Vec<_>>();
    if search_types.is_empty() {
        return tool_error(
            "types must contain one or more of: album, artist, playlist, track, show, episode, audiobook",
        );
    }
    match client.search(
        &query,
        &search_types,
        coerce_limit(args.get("limit"), 10, 1, 50),
        optional_i64(args.get("offset")).unwrap_or(0).max(0),
        optional_string(args.get("market")),
        optional_string(args.get("include_external")),
    ) {
        Ok(payload) => tool_result(payload),
        Err(error) => spotify_tool_error(error),
    }
}

pub fn handle_spotify_playlists(args: &Value, runtime: &ToolRuntime) -> String {
    let action = lower_string(args.get("action")).unwrap_or_else(|| "list".to_string());
    let client = match spotify_client(runtime.hermes_home()) {
        Ok(client) => client,
        Err(error) => return spotify_tool_error(error),
    };
    let result = match action.as_str() {
        "list" => client.get_my_playlists(
            coerce_limit(args.get("limit"), 20, 1, 50),
            optional_i64(args.get("offset")).unwrap_or(0).max(0),
        ),
        "get" => {
            let playlist_id = match normalize_spotify_id(
                optional_string(args.get("playlist_id"))
                    .unwrap_or_default()
                    .as_str(),
                Some("playlist"),
            ) {
                Ok(value) => value,
                Err(error) => return tool_error(error),
            };
            client.get_playlist(&playlist_id, optional_string(args.get("market")))
        }
        "create" => {
            let Some(name) = optional_string(args.get("name")) else {
                return tool_error("name is required for action='create'");
            };
            client.create_playlist(
                &name,
                coerce_bool(args.get("public"), false),
                coerce_bool(args.get("collaborative"), false),
                optional_string(args.get("description")),
            )
        }
        "add_items" => {
            let playlist_id = match normalize_spotify_id(
                optional_string(args.get("playlist_id"))
                    .unwrap_or_default()
                    .as_str(),
                Some("playlist"),
            ) {
                Ok(value) => value,
                Err(error) => return tool_error(error),
            };
            let uris = match normalize_spotify_uris(as_list(args.get("uris")), None) {
                Ok(value) => value,
                Err(error) => return tool_error(error),
            };
            client.add_playlist_items(&playlist_id, &uris, optional_i64(args.get("position")))
        }
        "remove_items" => {
            let playlist_id = match normalize_spotify_id(
                optional_string(args.get("playlist_id"))
                    .unwrap_or_default()
                    .as_str(),
                Some("playlist"),
            ) {
                Ok(value) => value,
                Err(error) => return tool_error(error),
            };
            let uris = match normalize_spotify_uris(as_list(args.get("uris")), None) {
                Ok(value) => value,
                Err(error) => return tool_error(error),
            };
            client.remove_playlist_items(
                &playlist_id,
                &uris,
                optional_string(args.get("snapshot_id")),
            )
        }
        "update_details" => {
            let playlist_id = match normalize_spotify_id(
                optional_string(args.get("playlist_id"))
                    .unwrap_or_default()
                    .as_str(),
                Some("playlist"),
            ) {
                Ok(value) => value,
                Err(error) => return tool_error(error),
            };
            client.update_playlist_details(
                &playlist_id,
                optional_string(args.get("name")),
                optional_bool(args.get("public")),
                optional_bool(args.get("collaborative")),
                optional_string(args.get("description")),
            )
        }
        _ => return tool_error(format!("Unknown spotify_playlists action: {action}")),
    };
    match result {
        Ok(payload) => tool_result(payload),
        Err(error) => spotify_tool_error(error),
    }
}

pub fn handle_spotify_albums(args: &Value, runtime: &ToolRuntime) -> String {
    let action = lower_string(args.get("action")).unwrap_or_else(|| "get".to_string());
    let client = match spotify_client(runtime.hermes_home()) {
        Ok(client) => client,
        Err(error) => return spotify_tool_error(error),
    };
    let raw_album = optional_string(args.get("album_id"))
        .or_else(|| optional_string(args.get("id")))
        .unwrap_or_default();
    let album_id = match normalize_spotify_id(&raw_album, Some("album")) {
        Ok(value) => value,
        Err(error) => return tool_error(error),
    };
    let result = match action.as_str() {
        "get" => client.get_album(&album_id, optional_string(args.get("market"))),
        "tracks" => client.get_album_tracks(
            &album_id,
            coerce_limit(args.get("limit"), 20, 1, 50),
            optional_i64(args.get("offset")).unwrap_or(0).max(0),
            optional_string(args.get("market")),
        ),
        _ => return tool_error(format!("Unknown spotify_albums action: {action}")),
    };
    match result {
        Ok(payload) => tool_result(payload),
        Err(error) => spotify_tool_error(error),
    }
}

pub fn handle_spotify_library(args: &Value, runtime: &ToolRuntime) -> String {
    let Some(kind) = lower_string(args.get("kind")) else {
        return tool_error("kind must be one of: tracks, albums");
    };
    if !matches!(kind.as_str(), "tracks" | "albums") {
        return tool_error("kind must be one of: tracks, albums");
    }
    let action = lower_string(args.get("action")).unwrap_or_else(|| "list".to_string());
    let client = match spotify_client(runtime.hermes_home()) {
        Ok(client) => client,
        Err(error) => return spotify_tool_error(error),
    };
    let result = match action.as_str() {
        "list" => {
            let limit = coerce_limit(args.get("limit"), 20, 1, 50);
            let offset = optional_i64(args.get("offset")).unwrap_or(0).max(0);
            let market = optional_string(args.get("market"));
            if kind == "tracks" {
                client.get_saved_tracks(limit, offset, market)
            } else {
                client.get_saved_albums(limit, offset, market)
            }
        }
        "save" => {
            let values = if args.get("uris").is_some() {
                as_list(args.get("uris"))
            } else {
                as_list(args.get("items"))
            };
            let item_type = if kind == "tracks" { "track" } else { "album" };
            let uris = match normalize_spotify_uris(values, Some(item_type)) {
                Ok(value) => value,
                Err(error) => return tool_error(error),
            };
            client.save_library_items(&uris)
        }
        "remove" => {
            let values = if args.get("ids").is_some() {
                as_list(args.get("ids"))
            } else {
                as_list(args.get("items"))
            };
            if values.is_empty() {
                return tool_error("ids/items is required for action='remove'");
            }
            let ids = values
                .iter()
                .map(|item| {
                    normalize_spotify_id(
                        item,
                        Some(if kind == "tracks" { "track" } else { "album" }),
                    )
                })
                .collect::<Result<Vec<_>, _>>();
            let ids = match ids {
                Ok(value) => value,
                Err(error) => return tool_error(error),
            };
            if kind == "tracks" {
                client.remove_saved_tracks(&ids)
            } else {
                client.remove_saved_albums(&ids)
            }
        }
        _ => return tool_error(format!("Unknown spotify_library action: {action}")),
    };
    match result {
        Ok(payload) => tool_result(payload),
        Err(error) => spotify_tool_error(error),
    }
}

impl SpotifyClient {
    fn new(home: &Path) -> Result<Self, SpotifyError> {
        let runtime = resolve_spotify_runtime_credentials(
            home,
            false,
            true,
            SPOTIFY_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
        )?;
        Ok(Self {
            home: home.to_path_buf(),
            runtime,
        })
    }

    fn request(
        &mut self,
        method: Method,
        path: &str,
        params: Vec<(String, String)>,
        json_body: Option<Value>,
        allow_retry_on_401: bool,
        empty_response: Option<Value>,
    ) -> Result<Value, SpotifyError> {
        let url = format!("{}{}", self.runtime.base_url, path);
        let client = Client::builder()
            .timeout(Duration::from_secs(SPOTIFY_REQUEST_TIMEOUT_SECS))
            .build()
            .map_err(|error| SpotifyError::Message(format!("Spotify tool failed: {error}")))?;
        let mut request = client
            .request(method.clone(), url)
            .header(
                "Authorization",
                format!("{} {}", self.runtime.token_type, self.runtime.access_token),
            )
            .header("Content-Type", "application/json");
        if !params.is_empty() {
            request = request.query(&params);
        }
        if let Some(body) = json_body.as_ref() {
            request = request.json(&strip_nulls(body.clone()));
        }

        let response = request
            .send()
            .map_err(|error| SpotifyError::Message(format!("Spotify request failed: {error}")))?;
        if response.status().as_u16() == 401 && allow_retry_on_401 {
            self.runtime = resolve_spotify_runtime_credentials(
                &self.home,
                true,
                true,
                SPOTIFY_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
            )?;
            return self.request(method, path, params, json_body, false, empty_response);
        }
        if response.status().as_u16() >= 400 {
            return Err(SpotifyError::Api(spotify_api_error(response, path)));
        }
        if response.status().as_u16() == 204 {
            return Ok(empty_response.unwrap_or_else(|| {
                json!({
                    "success": true,
                    "status_code": 204,
                    "empty": true
                })
            }));
        }

        let status_code = response.status().as_u16();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let text = response.text().map_err(|error| {
            SpotifyError::Message(format!("Failed to read Spotify response: {error}"))
        })?;
        if text.is_empty() {
            return Ok(empty_response.unwrap_or_else(|| {
                json!({
                    "success": true,
                    "status_code": status_code,
                    "empty": true
                })
            }));
        }
        if content_type.contains("application/json") {
            return serde_json::from_str::<Value>(&text).map_err(|error| {
                SpotifyError::Message(format!("Failed to parse Spotify JSON response: {error}"))
            });
        }
        Ok(json!({
            "success": true,
            "text": text,
        }))
    }

    fn get_devices(mut self) -> Result<Value, SpotifyError> {
        self.request(
            Method::GET,
            "/me/player/devices",
            Vec::new(),
            None,
            true,
            None,
        )
    }

    fn transfer_playback(mut self, device_id: &str, play: bool) -> Result<Value, SpotifyError> {
        self.request(
            Method::PUT,
            "/me/player",
            Vec::new(),
            Some(json!({
                "device_ids": [device_id],
                "play": play,
            })),
            true,
            None,
        )
    }

    fn get_playback_state(mut self, market: Option<String>) -> Result<Value, SpotifyError> {
        let mut params = Vec::new();
        if let Some(market) = market {
            params.push(("market".to_string(), market));
        }
        self.request(
            Method::GET,
            "/me/player",
            params,
            None,
            true,
            Some(json!({
                "status_code": 204,
                "empty": true,
                "message": "No active Spotify playback session was found. Open Spotify on a device and start playback, or transfer playback to an available device."
            })),
        )
    }

    fn get_currently_playing(mut self, market: Option<String>) -> Result<Value, SpotifyError> {
        let mut params = Vec::new();
        if let Some(market) = market {
            params.push(("market".to_string(), market));
        }
        self.request(
            Method::GET,
            "/me/player/currently-playing",
            params,
            None,
            true,
            Some(json!({
                "status_code": 204,
                "empty": true,
                "message": "Spotify is not currently playing anything. Start playback in Spotify and try again."
            })),
        )
    }

    fn start_playback(
        mut self,
        device_id: Option<String>,
        context_uri: Option<String>,
        uris: Option<Vec<String>>,
        offset: Option<Value>,
        position_ms: Option<i64>,
    ) -> Result<Value, SpotifyError> {
        let mut params = Vec::new();
        if let Some(device_id) = device_id {
            params.push(("device_id".to_string(), device_id));
        }
        let mut body = Map::new();
        if let Some(context_uri) = context_uri {
            body.insert("context_uri".to_string(), Value::String(context_uri));
        }
        if let Some(uris) = uris {
            body.insert(
                "uris".to_string(),
                Value::Array(uris.into_iter().map(Value::String).collect()),
            );
        }
        if let Some(offset) = offset {
            body.insert("offset".to_string(), offset);
        }
        if let Some(position_ms) = position_ms {
            body.insert("position_ms".to_string(), json!(position_ms));
        }
        self.request(
            Method::PUT,
            "/me/player/play",
            params,
            Some(Value::Object(body)),
            true,
            None,
        )
    }

    fn pause_playback(mut self, device_id: Option<String>) -> Result<Value, SpotifyError> {
        let params = optional_param("device_id", device_id);
        self.request(Method::PUT, "/me/player/pause", params, None, true, None)
    }

    fn skip_next(mut self, device_id: Option<String>) -> Result<Value, SpotifyError> {
        let params = optional_param("device_id", device_id);
        self.request(Method::POST, "/me/player/next", params, None, true, None)
    }

    fn skip_previous(mut self, device_id: Option<String>) -> Result<Value, SpotifyError> {
        let params = optional_param("device_id", device_id);
        self.request(
            Method::POST,
            "/me/player/previous",
            params,
            None,
            true,
            None,
        )
    }

    fn seek(mut self, position_ms: i64, device_id: Option<String>) -> Result<Value, SpotifyError> {
        let mut params = vec![("position_ms".to_string(), position_ms.to_string())];
        if let Some(device_id) = device_id {
            params.push(("device_id".to_string(), device_id));
        }
        self.request(Method::PUT, "/me/player/seek", params, None, true, None)
    }

    fn set_repeat(mut self, state: &str, device_id: Option<String>) -> Result<Value, SpotifyError> {
        let mut params = vec![("state".to_string(), state.to_string())];
        if let Some(device_id) = device_id {
            params.push(("device_id".to_string(), device_id));
        }
        self.request(Method::PUT, "/me/player/repeat", params, None, true, None)
    }

    fn set_shuffle(
        mut self,
        state: bool,
        device_id: Option<String>,
    ) -> Result<Value, SpotifyError> {
        let mut params = vec![("state".to_string(), state.to_string())];
        if let Some(device_id) = device_id {
            params.push(("device_id".to_string(), device_id));
        }
        self.request(Method::PUT, "/me/player/shuffle", params, None, true, None)
    }

    fn set_volume(
        mut self,
        volume_percent: i64,
        device_id: Option<String>,
    ) -> Result<Value, SpotifyError> {
        let mut params = vec![("volume_percent".to_string(), volume_percent.to_string())];
        if let Some(device_id) = device_id {
            params.push(("device_id".to_string(), device_id));
        }
        self.request(Method::PUT, "/me/player/volume", params, None, true, None)
    }

    fn get_queue(mut self) -> Result<Value, SpotifyError> {
        self.request(
            Method::GET,
            "/me/player/queue",
            Vec::new(),
            None,
            true,
            None,
        )
    }

    fn add_to_queue(mut self, uri: &str, device_id: Option<String>) -> Result<Value, SpotifyError> {
        let mut params = vec![("uri".to_string(), uri.to_string())];
        if let Some(device_id) = device_id {
            params.push(("device_id".to_string(), device_id));
        }
        self.request(Method::POST, "/me/player/queue", params, None, true, None)
    }

    fn search(
        mut self,
        query: &str,
        search_types: &[String],
        limit: i64,
        offset: i64,
        market: Option<String>,
        include_external: Option<String>,
    ) -> Result<Value, SpotifyError> {
        let mut params = vec![
            ("q".to_string(), query.to_string()),
            ("type".to_string(), search_types.join(",")),
            ("limit".to_string(), limit.to_string()),
            ("offset".to_string(), offset.to_string()),
        ];
        if let Some(market) = market {
            params.push(("market".to_string(), market));
        }
        if let Some(include_external) = include_external {
            params.push(("include_external".to_string(), include_external));
        }
        self.request(Method::GET, "/search", params, None, true, None)
    }

    fn get_my_playlists(mut self, limit: i64, offset: i64) -> Result<Value, SpotifyError> {
        self.request(
            Method::GET,
            "/me/playlists",
            vec![
                ("limit".to_string(), limit.to_string()),
                ("offset".to_string(), offset.to_string()),
            ],
            None,
            true,
            None,
        )
    }

    fn get_playlist(
        mut self,
        playlist_id: &str,
        market: Option<String>,
    ) -> Result<Value, SpotifyError> {
        self.request(
            Method::GET,
            &format!("/playlists/{playlist_id}"),
            optional_param("market", market),
            None,
            true,
            None,
        )
    }

    fn create_playlist(
        mut self,
        name: &str,
        public: bool,
        collaborative: bool,
        description: Option<String>,
    ) -> Result<Value, SpotifyError> {
        self.request(
            Method::POST,
            "/me/playlists",
            Vec::new(),
            Some(json!({
                "name": name,
                "public": public,
                "collaborative": collaborative,
                "description": description,
            })),
            true,
            None,
        )
    }

    fn add_playlist_items(
        mut self,
        playlist_id: &str,
        uris: &[String],
        position: Option<i64>,
    ) -> Result<Value, SpotifyError> {
        self.request(
            Method::POST,
            &format!("/playlists/{playlist_id}/items"),
            Vec::new(),
            Some(json!({
                "uris": uris,
                "position": position,
            })),
            true,
            None,
        )
    }

    fn remove_playlist_items(
        mut self,
        playlist_id: &str,
        uris: &[String],
        snapshot_id: Option<String>,
    ) -> Result<Value, SpotifyError> {
        let items = uris
            .iter()
            .map(|uri| json!({ "uri": uri }))
            .collect::<Vec<_>>();
        self.request(
            Method::DELETE,
            &format!("/playlists/{playlist_id}/items"),
            Vec::new(),
            Some(json!({
                "items": items,
                "snapshot_id": snapshot_id,
            })),
            true,
            None,
        )
    }

    fn update_playlist_details(
        mut self,
        playlist_id: &str,
        name: Option<String>,
        public: Option<bool>,
        collaborative: Option<bool>,
        description: Option<String>,
    ) -> Result<Value, SpotifyError> {
        self.request(
            Method::PUT,
            &format!("/playlists/{playlist_id}"),
            Vec::new(),
            Some(json!({
                "name": name,
                "public": public,
                "collaborative": collaborative,
                "description": description,
            })),
            true,
            None,
        )
    }

    fn get_album(mut self, album_id: &str, market: Option<String>) -> Result<Value, SpotifyError> {
        self.request(
            Method::GET,
            &format!("/albums/{album_id}"),
            optional_param("market", market),
            None,
            true,
            None,
        )
    }

    fn get_album_tracks(
        mut self,
        album_id: &str,
        limit: i64,
        offset: i64,
        market: Option<String>,
    ) -> Result<Value, SpotifyError> {
        let mut params = vec![
            ("limit".to_string(), limit.to_string()),
            ("offset".to_string(), offset.to_string()),
        ];
        if let Some(market) = market {
            params.push(("market".to_string(), market));
        }
        self.request(
            Method::GET,
            &format!("/albums/{album_id}/tracks"),
            params,
            None,
            true,
            None,
        )
    }

    fn get_saved_tracks(
        mut self,
        limit: i64,
        offset: i64,
        market: Option<String>,
    ) -> Result<Value, SpotifyError> {
        let mut params = vec![
            ("limit".to_string(), limit.to_string()),
            ("offset".to_string(), offset.to_string()),
        ];
        if let Some(market) = market {
            params.push(("market".to_string(), market));
        }
        self.request(Method::GET, "/me/tracks", params, None, true, None)
    }

    fn save_library_items(mut self, uris: &[String]) -> Result<Value, SpotifyError> {
        self.request(
            Method::PUT,
            "/me/library",
            vec![("uris".to_string(), uris.join(","))],
            None,
            true,
            None,
        )
    }

    fn get_saved_albums(
        mut self,
        limit: i64,
        offset: i64,
        market: Option<String>,
    ) -> Result<Value, SpotifyError> {
        let mut params = vec![
            ("limit".to_string(), limit.to_string()),
            ("offset".to_string(), offset.to_string()),
        ];
        if let Some(market) = market {
            params.push(("market".to_string(), market));
        }
        self.request(Method::GET, "/me/albums", params, None, true, None)
    }

    fn remove_saved_tracks(mut self, track_ids: &[String]) -> Result<Value, SpotifyError> {
        let uris = track_ids
            .iter()
            .map(|id| format!("spotify:track:{id}"))
            .collect::<Vec<_>>();
        self.request(
            Method::DELETE,
            "/me/library",
            vec![("uris".to_string(), uris.join(","))],
            None,
            true,
            None,
        )
    }

    fn remove_saved_albums(mut self, album_ids: &[String]) -> Result<Value, SpotifyError> {
        let uris = album_ids
            .iter()
            .map(|id| format!("spotify:album:{id}"))
            .collect::<Vec<_>>();
        self.request(
            Method::DELETE,
            "/me/library",
            vec![("uris".to_string(), uris.join(","))],
            None,
            true,
            None,
        )
    }

    fn get_recently_played(
        mut self,
        limit: i64,
        after: Option<i64>,
        before: Option<i64>,
    ) -> Result<Value, SpotifyError> {
        let mut params = vec![("limit".to_string(), limit.to_string())];
        if let Some(after) = after {
            params.push(("after".to_string(), after.to_string()));
        }
        if let Some(before) = before {
            params.push(("before".to_string(), before.to_string()));
        }
        self.request(
            Method::GET,
            "/me/player/recently-played",
            params,
            None,
            true,
            None,
        )
    }
}

fn spotify_client(home: &Path) -> Result<SpotifyClient, SpotifyError> {
    SpotifyClient::new(home)
}

fn resolve_spotify_runtime_credentials(
    home: &Path,
    force_refresh: bool,
    refresh_if_expiring: bool,
    refresh_skew_seconds: i64,
) -> Result<SpotifyRuntimeCredentials, SpotifyError> {
    let mut auth_store = load_auth_store(home)?;
    let mut state = load_provider_state(&auth_store, "spotify").ok_or_else(|| {
        SpotifyError::Message(
            "Spotify is not authenticated. Run `hermes auth spotify` first.".to_string(),
        )
    })?;

    let should_refresh = force_refresh
        || (refresh_if_expiring && is_expiring(state.get("expires_at"), refresh_skew_seconds));
    if should_refresh {
        state = refresh_spotify_oauth_state(&state)?;
        store_provider_state(&mut auth_store, "spotify", state.clone())?;
        save_auth_store(home, &auth_store)?;
    }

    let access_token = map_string(&state, "access_token").unwrap_or_default();
    if access_token.is_empty() {
        return Err(SpotifyError::Message(
            "Spotify access token missing. Run `hermes auth spotify` again.".to_string(),
        ));
    }

    Ok(SpotifyRuntimeCredentials {
        access_token,
        token_type: map_string(&state, "token_type").unwrap_or_else(|| "Bearer".to_string()),
        base_url: spotify_api_base_url(Some(&state)),
    })
}

fn refresh_spotify_oauth_state(
    state: &Map<String, Value>,
) -> Result<Map<String, Value>, SpotifyError> {
    let refresh_token = map_string(state, "refresh_token").unwrap_or_default();
    if refresh_token.is_empty() {
        return Err(SpotifyError::Message(
            "Spotify refresh token missing. Run `hermes auth spotify` again.".to_string(),
        ));
    }

    let client_id = spotify_client_id(Some(state))?;
    let accounts_base_url = spotify_accounts_base_url(Some(state));
    let client = Client::builder()
        .timeout(Duration::from_secs(SPOTIFY_REFRESH_TIMEOUT_SECS))
        .build()
        .map_err(|error| SpotifyError::Message(format!("Spotify token refresh failed: {error}")))?;
    let response = client
        .post(format!("{accounts_base_url}/api/token"))
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(&[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token.as_str()),
            ("client_id", client_id.as_str()),
        ])
        .send()
        .map_err(|error| SpotifyError::Message(format!("Spotify token refresh failed: {error}")))?;

    let status = response.status().as_u16();
    let text = response
        .text()
        .map_err(|error| SpotifyError::Message(format!("Spotify token refresh failed: {error}")))?;
    if status >= 400 {
        return Err(SpotifyError::Message(format!(
            "Spotify token refresh failed. Run `hermes auth spotify` again.{}",
            if text.trim().is_empty() {
                String::new()
            } else {
                format!(" Response: {}", text.trim())
            }
        )));
    }
    let payload = serde_json::from_str::<Value>(&text)
        .map_err(|error| SpotifyError::Message(format!("Spotify token refresh failed: {error}")))?;
    let payload = payload.as_object().ok_or_else(|| {
        SpotifyError::Message(
            "Spotify refresh response did not include an access_token.".to_string(),
        )
    })?;
    if map_string(payload, "access_token")
        .unwrap_or_default()
        .is_empty()
    {
        return Err(SpotifyError::Message(
            "Spotify refresh response did not include an access_token.".to_string(),
        ));
    }

    Ok(spotify_token_payload_to_state(
        payload,
        &client_id,
        &spotify_redirect_uri(Some(state)),
        map_string(state, "scope").unwrap_or_else(|| DEFAULT_SPOTIFY_SCOPE.to_string()),
        &accounts_base_url,
        &spotify_api_base_url(Some(state)),
        Some(state),
    ))
}

fn spotify_token_payload_to_state(
    token_payload: &Map<String, Value>,
    client_id: &str,
    redirect_uri: &str,
    requested_scope: String,
    accounts_base_url: &str,
    api_base_url: &str,
    previous_state: Option<&Map<String, Value>>,
) -> Map<String, Value> {
    let now = Utc::now();
    let expires_in = coerce_ttl_seconds(token_payload.get("expires_in"));
    let expires_at = now + chrono::Duration::seconds(expires_in as i64);

    let mut state = previous_state.cloned().unwrap_or_default();
    state.insert(
        "client_id".to_string(),
        Value::String(client_id.to_string()),
    );
    state.insert(
        "redirect_uri".to_string(),
        Value::String(redirect_uri.to_string()),
    );
    state.insert(
        "accounts_base_url".to_string(),
        Value::String(accounts_base_url.to_string()),
    );
    state.insert(
        "api_base_url".to_string(),
        Value::String(api_base_url.to_string()),
    );
    state.insert("scope".to_string(), Value::String(requested_scope.clone()));
    state.insert(
        "granted_scope".to_string(),
        Value::String(map_string(token_payload, "scope").unwrap_or(requested_scope.clone())),
    );
    state.insert(
        "token_type".to_string(),
        Value::String(
            map_string(token_payload, "token_type").unwrap_or_else(|| "Bearer".to_string()),
        ),
    );
    state.insert(
        "access_token".to_string(),
        Value::String(map_string(token_payload, "access_token").unwrap_or_default()),
    );
    let refresh_token = map_string(token_payload, "refresh_token")
        .or_else(|| previous_state.and_then(|state| map_string(state, "refresh_token")))
        .unwrap_or_default();
    state.insert("refresh_token".to_string(), Value::String(refresh_token));
    state.insert("obtained_at".to_string(), Value::String(now.to_rfc3339()));
    state.insert(
        "expires_at".to_string(),
        Value::String(expires_at.to_rfc3339()),
    );
    state.insert("expires_in".to_string(), json!(expires_in));
    state.insert(
        "auth_type".to_string(),
        Value::String("oauth_pkce".to_string()),
    );
    state
}

fn spotify_auth_logged_in(home: &Path) -> bool {
    let Ok(store) = load_auth_store(home) else {
        return false;
    };
    let Some(state) = load_provider_state(&store, "spotify") else {
        return false;
    };
    let refresh_token = map_string(&state, "refresh_token").unwrap_or_default();
    !refresh_token.is_empty() || !is_expiring(state.get("expires_at"), 0)
}

fn load_auth_store(home: &Path) -> Result<Value, SpotifyError> {
    let path = home.join("auth.json");
    if !path.exists() {
        return Ok(json!({
            "version": 1,
            "providers": {}
        }));
    }
    let text = fs::read_to_string(&path).map_err(|error| {
        SpotifyError::Message(format!("Failed to read {}: {error}", path.display()))
    })?;
    serde_json::from_str::<Value>(&text).map_err(|error| {
        SpotifyError::Message(format!("Failed to parse {}: {error}", path.display()))
    })
}

fn save_auth_store(home: &Path, auth_store: &Value) -> Result<(), SpotifyError> {
    let path = home.join("auth.json");
    let payload = serde_json::to_string_pretty(auth_store).map_err(|error| {
        SpotifyError::Message(format!("Failed to encode {}: {error}", path.display()))
    })?;
    fs::write(&path, format!("{payload}\n")).map_err(|error| {
        SpotifyError::Message(format!("Failed to write {}: {error}", path.display()))
    })
}

fn load_provider_state(auth_store: &Value, provider_id: &str) -> Option<Map<String, Value>> {
    auth_store
        .get("providers")
        .and_then(Value::as_object)
        .and_then(|providers| providers.get(provider_id))
        .and_then(Value::as_object)
        .cloned()
}

fn store_provider_state(
    auth_store: &mut Value,
    provider_id: &str,
    state: Map<String, Value>,
) -> Result<(), SpotifyError> {
    let root = auth_store.as_object_mut().ok_or_else(|| {
        SpotifyError::Message("Spotify auth store is not a JSON object.".to_string())
    })?;
    let providers = root
        .entry("providers".to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    let providers = providers.as_object_mut().ok_or_else(|| {
        SpotifyError::Message("Spotify auth store providers field is invalid.".to_string())
    })?;
    providers.insert(provider_id.to_string(), Value::Object(state));
    Ok(())
}

fn spotify_client_id(state: Option<&Map<String, Value>>) -> Result<String, SpotifyError> {
    for candidate in [
        env::var("HERMES_SPOTIFY_CLIENT_ID").ok(),
        env::var("SPOTIFY_CLIENT_ID").ok(),
        state.and_then(|state| map_string(state, "client_id")),
    ] {
        let cleaned = candidate.unwrap_or_default().trim().to_string();
        if !cleaned.is_empty() {
            return Ok(cleaned);
        }
    }
    Err(SpotifyError::Message(
        "Spotify client_id is required. Set HERMES_SPOTIFY_CLIENT_ID or pass --client-id."
            .to_string(),
    ))
}

fn spotify_redirect_uri(state: Option<&Map<String, Value>>) -> String {
    for candidate in [
        env::var("HERMES_SPOTIFY_REDIRECT_URI").ok(),
        env::var("SPOTIFY_REDIRECT_URI").ok(),
        state.and_then(|state| map_string(state, "redirect_uri")),
        Some(DEFAULT_SPOTIFY_REDIRECT_URI.to_string()),
    ] {
        let cleaned = candidate.unwrap_or_default().trim().to_string();
        if !cleaned.is_empty() {
            return cleaned;
        }
    }
    DEFAULT_SPOTIFY_REDIRECT_URI.to_string()
}

fn spotify_api_base_url(state: Option<&Map<String, Value>>) -> String {
    for candidate in [
        env::var("HERMES_SPOTIFY_API_BASE_URL").ok(),
        state.and_then(|state| map_string(state, "api_base_url")),
        Some(DEFAULT_SPOTIFY_API_BASE_URL.to_string()),
    ] {
        let cleaned = candidate
            .unwrap_or_default()
            .trim()
            .trim_end_matches('/')
            .to_string();
        if !cleaned.is_empty() {
            return cleaned;
        }
    }
    DEFAULT_SPOTIFY_API_BASE_URL.to_string()
}

fn spotify_accounts_base_url(state: Option<&Map<String, Value>>) -> String {
    for candidate in [
        env::var("HERMES_SPOTIFY_ACCOUNTS_BASE_URL").ok(),
        state.and_then(|state| map_string(state, "accounts_base_url")),
        Some(DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL.to_string()),
    ] {
        let cleaned = candidate
            .unwrap_or_default()
            .trim()
            .trim_end_matches('/')
            .to_string();
        if !cleaned.is_empty() {
            return cleaned;
        }
    }
    DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL.to_string()
}

fn spotify_api_error(response: reqwest::blocking::Response, path: &str) -> SpotifyApiError {
    let status_code = response.status().as_u16();
    let retry_after = response
        .headers()
        .get("Retry-After")
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned);
    let text = response.text().unwrap_or_default();
    let detail = extract_spotify_error_detail(&text);
    SpotifyApiError {
        message: friendly_spotify_error_message(status_code, &detail, path, retry_after.as_deref()),
        status_code: Some(status_code),
        response_body: (!text.trim().is_empty()).then_some(text),
    }
}

fn extract_spotify_error_detail(text: &str) -> String {
    serde_json::from_str::<Value>(text)
        .ok()
        .and_then(|payload| {
            if let Some(error) = payload.get("error") {
                if let Some(error_obj) = error.as_object() {
                    return map_string(error_obj, "message");
                }
                return error.as_str().map(ToOwned::to_owned);
            }
            None
        })
        .unwrap_or_else(|| text.trim().to_string())
}

fn friendly_spotify_error_message(
    status_code: u16,
    detail: &str,
    path: &str,
    retry_after: Option<&str>,
) -> String {
    let normalized_detail = detail.to_ascii_lowercase();
    let is_playback_path = path.starts_with("/me/player");

    if status_code == 401 {
        return "Spotify authentication failed or expired. Run `hermes auth spotify` again."
            .to_string();
    }
    if status_code == 403 {
        if is_playback_path {
            return "Spotify rejected this playback request. Playback control usually requires a Spotify Premium account and an active Spotify Connect device.".to_string();
        }
        if normalized_detail.contains("scope") || normalized_detail.contains("permission") {
            return "Spotify rejected the request because the current auth scope is insufficient. Re-run `hermes auth spotify` to refresh permissions.".to_string();
        }
        return "Spotify rejected the request. The account may not have permission for this action.".to_string();
    }
    if status_code == 404 {
        if is_playback_path {
            return "Spotify could not find an active playback device or player session for this request.".to_string();
        }
        return "Spotify resource not found.".to_string();
    }
    if status_code == 429 {
        let mut message = "Spotify rate limit exceeded.".to_string();
        if let Some(retry_after) = retry_after {
            message.push_str(&format!(" Retry after {retry_after} seconds."));
        }
        return message;
    }
    if !detail.trim().is_empty() {
        return detail.trim().to_string();
    }
    format!("Spotify API request failed with status {status_code}.")
}

fn describe_empty_playback(payload: &Value, action: &str) -> Option<Value> {
    if !payload
        .get("empty")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return None;
    }
    if action == "get_currently_playing" {
        return Some(json!({
            "success": true,
            "action": action,
            "is_playing": false,
            "status_code": payload.get("status_code").cloned().unwrap_or(json!(204)),
            "message": payload.get("message").cloned().unwrap_or_else(|| json!("Spotify is not currently playing anything.")),
        }));
    }
    if action == "get_state" {
        return Some(json!({
            "success": true,
            "action": action,
            "has_active_device": false,
            "status_code": payload.get("status_code").cloned().unwrap_or(json!(204)),
            "message": payload.get("message").cloned().unwrap_or_else(|| json!("No active Spotify playback session was found.")),
        }));
    }
    None
}

fn spotify_tool_error(error: SpotifyError) -> String {
    match error {
        SpotifyError::Message(message) => tool_error(message),
        SpotifyError::Api(error) => {
            let mut payload = json!({ "error": error.message });
            if let Some(status_code) = error.status_code {
                payload["status_code"] = json!(status_code);
            }
            if let Some(body) = error.response_body {
                payload["response_body"] = Value::String(body);
            }
            tool_result(payload)
        }
    }
}

fn coerce_limit(raw: Option<&Value>, default: i64, minimum: i64, maximum: i64) -> i64 {
    optional_i64(raw).unwrap_or(default).clamp(minimum, maximum)
}

fn coerce_bool(raw: Option<&Value>, default: bool) -> bool {
    match raw {
        Some(Value::Bool(value)) => *value,
        Some(Value::String(text)) => match text.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default,
        },
        _ => default,
    }
}

fn optional_bool(raw: Option<&Value>) -> Option<bool> {
    match raw {
        Some(Value::Bool(value)) => Some(*value),
        Some(Value::String(text)) => match text.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn as_list(raw: Option<&Value>) -> Vec<String> {
    match raw {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|item| match item {
                Value::String(text) => {
                    let cleaned = text.trim();
                    (!cleaned.is_empty()).then_some(cleaned.to_string())
                }
                other => {
                    let text = other.to_string();
                    let cleaned = text.trim_matches('"').trim().to_string();
                    (!cleaned.is_empty()).then_some(cleaned)
                }
            })
            .collect(),
        Some(Value::String(text)) => {
            let cleaned = text.trim();
            if cleaned.is_empty() {
                Vec::new()
            } else {
                vec![cleaned.to_string()]
            }
        }
        Some(other) => {
            let text = other.to_string();
            let cleaned = text.trim_matches('"').trim().to_string();
            if cleaned.is_empty() {
                Vec::new()
            } else {
                vec![cleaned]
            }
        }
        None => Vec::new(),
    }
}

fn optional_string(raw: Option<&Value>) -> Option<String> {
    match raw {
        Some(Value::String(text)) => {
            let cleaned = text.trim();
            (!cleaned.is_empty()).then_some(cleaned.to_string())
        }
        Some(Value::Number(number)) => Some(number.to_string()),
        Some(Value::Bool(boolean)) => Some(boolean.to_string()),
        _ => None,
    }
}

fn lower_string(raw: Option<&Value>) -> Option<String> {
    optional_string(raw).map(|value| value.to_ascii_lowercase())
}

fn optional_i64(raw: Option<&Value>) -> Option<i64> {
    match raw {
        Some(Value::Number(number)) => number.as_i64(),
        Some(Value::String(text)) => text.trim().parse::<i64>().ok(),
        _ => None,
    }
}

fn normalize_spotify_id(value: &str, expected_type: Option<&str>) -> Result<String, String> {
    let cleaned = value.trim();
    if cleaned.is_empty() {
        return Err("Spotify id/uri/url is required.".to_string());
    }
    if let Some(rest) = cleaned.strip_prefix("spotify:") {
        let parts = rest.split(':').collect::<Vec<_>>();
        if parts.len() >= 2 {
            let item_type = parts[0];
            if let Some(expected_type) = expected_type {
                if item_type != expected_type {
                    return Err(format!(
                        "Expected a Spotify {expected_type}, got {item_type}."
                    ));
                }
            }
            return Ok(parts[1].to_string());
        }
    }
    if cleaned.contains("open.spotify.com") {
        let parsed =
            Url::parse(cleaned).map_err(|_| "Spotify URI/url/id is required.".to_string())?;
        let path_parts = parsed
            .path_segments()
            .map(|parts| parts.filter(|part| !part.is_empty()).collect::<Vec<_>>())
            .unwrap_or_default();
        if path_parts.len() >= 2 {
            let item_type = path_parts[0];
            let item_id = path_parts[1];
            if let Some(expected_type) = expected_type {
                if item_type != expected_type {
                    return Err(format!(
                        "Expected a Spotify {expected_type}, got {item_type}."
                    ));
                }
            }
            return Ok(item_id.to_string());
        }
    }
    Ok(cleaned.to_string())
}

fn normalize_spotify_uri(value: &str, expected_type: Option<&str>) -> Result<String, String> {
    let cleaned = value.trim();
    if cleaned.is_empty() {
        return Err("Spotify URI/url/id is required.".to_string());
    }
    if cleaned.starts_with("spotify:") {
        if let Some(expected_type) = expected_type {
            let parts = cleaned.split(':').collect::<Vec<_>>();
            if parts.len() >= 3 && parts[1] != expected_type {
                return Err(format!(
                    "Expected a Spotify {expected_type}, got {}.",
                    parts[1]
                ));
            }
        }
        return Ok(cleaned.to_string());
    }
    let item_id = normalize_spotify_id(cleaned, expected_type)?;
    if let Some(expected_type) = expected_type {
        return Ok(format!("spotify:{expected_type}:{item_id}"));
    }
    Ok(cleaned.to_string())
}

fn normalize_spotify_uris(
    values: Vec<String>,
    expected_type: Option<&str>,
) -> Result<Vec<String>, String> {
    let mut uris = Vec::new();
    for value in values {
        let uri = normalize_spotify_uri(&value, expected_type)?;
        if !uris.contains(&uri) {
            uris.push(uri);
        }
    }
    if uris.is_empty() {
        return Err("At least one Spotify item is required.".to_string());
    }
    Ok(uris)
}

fn strip_nulls(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .filter_map(|(key, value)| {
                    if value.is_null() {
                        None
                    } else {
                        Some((key, strip_nulls(value)))
                    }
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.into_iter().map(strip_nulls).collect()),
        other => other,
    }
}

fn optional_param(key: &str, value: Option<String>) -> Vec<(String, String)> {
    value
        .map(|value| vec![(key.to_string(), value)])
        .unwrap_or_default()
}

fn map_string(map: &Map<String, Value>, key: &str) -> Option<String> {
    map.get(key).and_then(|value| match value {
        Value::String(text) => {
            let cleaned = text.trim();
            (!cleaned.is_empty()).then_some(cleaned.to_string())
        }
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(boolean) => Some(boolean.to_string()),
        _ => None,
    })
}

fn is_expiring(value: Option<&Value>, skew_seconds: i64) -> bool {
    let Some(Value::String(text)) = value else {
        return true;
    };
    let Ok(parsed) = DateTime::parse_from_rfc3339(text) else {
        return true;
    };
    parsed.with_timezone(&Utc) <= Utc::now() + chrono::Duration::seconds(skew_seconds)
}

fn coerce_ttl_seconds(value: Option<&Value>) -> i64 {
    value
        .and_then(|value| match value {
            Value::Number(number) => number.as_i64(),
            Value::String(text) => text.trim().parse::<i64>().ok(),
            _ => None,
        })
        .unwrap_or(0)
        .max(0)
}

fn hermes_home_from_env() -> PathBuf {
    env::var_os("HERMES_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".hermes")))
        .unwrap_or_else(|| PathBuf::from(".hermes"))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Mutex, OnceLock};
    use std::thread;

    use tempfile::TempDir;

    static TEST_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    fn test_env_lock() -> &'static Mutex<()> {
        TEST_ENV_LOCK.get_or_init(|| Mutex::new(()))
    }

    fn acquire_test_lock() -> std::sync::MutexGuard<'static, ()> {
        test_env_lock()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    fn with_env_var(key: &str, value: Option<&str>) {
        match value {
            Some(value) => unsafe { env::set_var(key, value) },
            None => unsafe { env::remove_var(key) },
        }
    }

    fn runtime_for(home: &TempDir) -> ToolRuntime {
        ToolRuntime::new(home.path()).with_hermes_home(home.path())
    }

    fn write_auth_store(home: &TempDir, state: Value) {
        fs::write(home.path().join("auth.json"), format!("{state}\n")).unwrap();
    }

    fn mock_server<F>(handler: F) -> (String, thread::JoinHandle<()>)
    where
        F: Fn(String, String) -> (u16, String, Vec<(&'static str, String)>) + Send + 'static,
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
                .map(|index| index + 4)
                .unwrap_or(request.len());
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
            let mut body_bytes = request[header_end..].to_vec();
            while body_bytes.len() < content_length {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                body_bytes.extend_from_slice(&buffer[..read]);
            }
            let body = String::from_utf8_lossy(&body_bytes).to_string();
            let (status, response_body, extra_headers) = handler(headers, body);
            let mut response = format!(
                "HTTP/1.1 {} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
                status,
                response_body.len()
            );
            for (key, value) in extra_headers {
                response.push_str(&format!("{key}: {value}\r\n"));
            }
            response.push_str("\r\n");
            response.push_str(&response_body);
            stream.write_all(response.as_bytes()).unwrap();
        });
        (format!("http://{addr}"), join)
    }

    #[test]
    fn spotify_toolset_is_opt_in() {
        assert_eq!(
            crate::resolve_toolset("spotify"),
            vec![
                "spotify_albums".to_string(),
                "spotify_devices".to_string(),
                "spotify_library".to_string(),
                "spotify_playback".to_string(),
                "spotify_playlists".to_string(),
                "spotify_queue".to_string(),
                "spotify_search".to_string(),
            ]
        );
        assert!(!crate::resolve_toolset("hermes-cli").contains(&"spotify_playback".to_string()));
    }

    #[test]
    fn normalize_spotify_uri_accepts_urls() {
        let uri = normalize_spotify_uri(
            "https://open.spotify.com/track/7ouMYWpwJ422jRcDASZB7P",
            Some("track"),
        )
        .unwrap();
        assert_eq!(uri, "spotify:track:7ouMYWpwJ422jRcDASZB7P");
    }

    #[test]
    fn spotify_available_uses_refresh_token_even_if_expired() {
        let _guard = acquire_test_lock();
        let home = TempDir::new().unwrap();
        write_auth_store(
            &home,
            json!({
                "version": 1,
                "providers": {
                    "spotify": {
                        "access_token": "expired",
                        "refresh_token": "refresh-token",
                        "expires_at": "2000-01-01T00:00:00+00:00"
                    }
                }
            }),
        );
        let old_home = env::var("HERMES_HOME").ok();
        with_env_var("HERMES_HOME", Some(home.path().to_str().unwrap()));
        assert!(spotify_available());
        with_env_var("HERMES_HOME", old_home.as_deref());
    }

    #[test]
    fn spotify_client_refreshes_expired_token_without_changing_active_provider() {
        let _guard = acquire_test_lock();
        let home = TempDir::new().unwrap();
        write_auth_store(
            &home,
            json!({
                "active_provider": "nous",
                "providers": {
                    "spotify": {
                        "client_id": "spotify-client",
                        "redirect_uri": DEFAULT_SPOTIFY_REDIRECT_URI,
                        "api_base_url": DEFAULT_SPOTIFY_API_BASE_URL,
                        "accounts_base_url": DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL,
                        "scope": DEFAULT_SPOTIFY_SCOPE,
                        "access_token": "expired-token",
                        "refresh_token": "refresh-token",
                        "token_type": "Bearer",
                        "expires_at": "2000-01-01T00:00:00+00:00"
                    }
                }
            }),
        );
        let (base_url, join) = mock_server(|headers, body| {
            assert!(headers.starts_with("POST /api/token "));
            assert!(body.contains("grant_type=refresh_token"));
            (
                200,
                json!({
                    "access_token": "fresh-token",
                    "token_type": "Bearer",
                    "expires_in": 3600
                })
                .to_string(),
                Vec::new(),
            )
        });

        let old_accounts = env::var("HERMES_SPOTIFY_ACCOUNTS_BASE_URL").ok();
        with_env_var("HERMES_SPOTIFY_ACCOUNTS_BASE_URL", Some(&base_url));
        let creds = resolve_spotify_runtime_credentials(home.path(), false, true, 120).unwrap();
        assert_eq!(creds.access_token, "fresh-token");

        let persisted = load_auth_store(home.path()).unwrap();
        assert_eq!(
            persisted["providers"]["spotify"]["access_token"],
            json!("fresh-token")
        );
        assert_eq!(persisted["active_provider"], json!("nous"));

        join.join().unwrap();
        with_env_var("HERMES_SPOTIFY_ACCOUNTS_BASE_URL", old_accounts.as_deref());
    }

    #[test]
    fn spotify_client_retries_once_after_401() {
        let _guard = acquire_test_lock();
        let home = TempDir::new().unwrap();
        write_auth_store(
            &home,
            json!({
                "providers": {
                    "spotify": {
                        "client_id": "spotify-client",
                        "redirect_uri": DEFAULT_SPOTIFY_REDIRECT_URI,
                        "api_base_url": "https://placeholder.invalid/v1",
                        "accounts_base_url": DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL,
                        "scope": DEFAULT_SPOTIFY_SCOPE,
                        "access_token": "token-1",
                        "refresh_token": "refresh-token",
                        "token_type": "Bearer",
                        "expires_at": "2099-01-01T00:00:00+00:00"
                    }
                }
            }),
        );

        let hit_counter = std::sync::Arc::new(Mutex::new(0_u32));
        let counter = hit_counter.clone();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let api_base_url = format!("http://{}", listener.local_addr().unwrap());
        let join_api = thread::spawn(move || {
            for _ in 0..2 {
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
                let headers = String::from_utf8_lossy(&request).to_string();
                let mut hits = counter.lock().unwrap();
                *hits += 1;
                let (status, response_body) = if *hits == 1 {
                    assert!(headers.starts_with("GET /v1/me/player/devices "));
                    (
                        401,
                        json!({"error": {"message": "expired token"}}).to_string(),
                    )
                } else {
                    assert!(headers.starts_with("GET /v1/me/player/devices "));
                    (200, json!({"devices": [{"id": "dev-1"}]}).to_string())
                };
                let response = format!(
                    "HTTP/1.1 {} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    status,
                    response_body.len(),
                    response_body
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        let (accounts_base_url, join_accounts) = mock_server(|headers, body| {
            assert!(headers.starts_with("POST /api/token "));
            assert!(body.contains("refresh_token=refresh-token"));
            (
                200,
                json!({
                    "access_token": "token-2",
                    "token_type": "Bearer",
                    "expires_in": 3600
                })
                .to_string(),
                Vec::new(),
            )
        });

        let old_api = env::var("HERMES_SPOTIFY_API_BASE_URL").ok();
        let old_accounts = env::var("HERMES_SPOTIFY_ACCOUNTS_BASE_URL").ok();
        with_env_var(
            "HERMES_SPOTIFY_API_BASE_URL",
            Some(&format!("{api_base_url}/v1")),
        );
        with_env_var("HERMES_SPOTIFY_ACCOUNTS_BASE_URL", Some(&accounts_base_url));

        let payload = spotify_client(home.path()).unwrap().get_devices().unwrap();
        assert_eq!(payload["devices"][0]["id"], json!("dev-1"));

        join_api.join().unwrap();
        join_accounts.join().unwrap();
        with_env_var("HERMES_SPOTIFY_API_BASE_URL", old_api.as_deref());
        with_env_var("HERMES_SPOTIFY_ACCOUNTS_BASE_URL", old_accounts.as_deref());
    }

    #[test]
    fn spotify_playback_empty_currently_playing_is_explanatory() {
        let payload = describe_empty_playback(
            &json!({
                "status_code": 204,
                "empty": true,
                "message": "Spotify is not currently playing anything. Start playback in Spotify and try again."
            }),
            "get_currently_playing",
        )
        .unwrap();
        assert_eq!(
            payload,
            json!({
                "success": true,
                "action": "get_currently_playing",
                "is_playing": false,
                "status_code": 204,
                "message": "Spotify is not currently playing anything. Start playback in Spotify and try again."
            })
        );
    }

    #[test]
    fn spotify_library_tracks_list_routes_to_saved_tracks() {
        let _guard = acquire_test_lock();
        let home = TempDir::new().unwrap();
        write_auth_store(
            &home,
            json!({
                "providers": {
                    "spotify": {
                        "client_id": "spotify-client",
                        "redirect_uri": DEFAULT_SPOTIFY_REDIRECT_URI,
                        "api_base_url": "https://placeholder.invalid/v1",
                        "accounts_base_url": DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL,
                        "scope": DEFAULT_SPOTIFY_SCOPE,
                        "access_token": "token-1",
                        "token_type": "Bearer",
                        "expires_at": "2099-01-01T00:00:00+00:00"
                    }
                }
            }),
        );
        let (base_url, join) = mock_server(|headers, _body| {
            assert!(headers.starts_with("GET /v1/me/tracks?limit=20&offset=0 "));
            (
                200,
                json!({"items": [], "total": 0}).to_string(),
                Vec::new(),
            )
        });
        let old_api = env::var("HERMES_SPOTIFY_API_BASE_URL").ok();
        with_env_var(
            "HERMES_SPOTIFY_API_BASE_URL",
            Some(&format!("{base_url}/v1")),
        );

        let payload = serde_json::from_str::<Value>(&handle_spotify_library(
            &json!({"kind": "tracks", "action": "list"}),
            &runtime_for(&home),
        ))
        .unwrap();
        assert_eq!(payload["total"], json!(0));

        join.join().unwrap();
        with_env_var("HERMES_SPOTIFY_API_BASE_URL", old_api.as_deref());
    }

    #[test]
    fn spotify_playback_403_uses_friendly_message() {
        let _guard = acquire_test_lock();
        let home = TempDir::new().unwrap();
        write_auth_store(
            &home,
            json!({
                "providers": {
                    "spotify": {
                        "client_id": "spotify-client",
                        "redirect_uri": DEFAULT_SPOTIFY_REDIRECT_URI,
                        "api_base_url": "https://placeholder.invalid/v1",
                        "accounts_base_url": DEFAULT_SPOTIFY_ACCOUNTS_BASE_URL,
                        "scope": DEFAULT_SPOTIFY_SCOPE,
                        "access_token": "token-1",
                        "token_type": "Bearer",
                        "expires_at": "2099-01-01T00:00:00+00:00"
                    }
                }
            }),
        );
        let (base_url, join) = mock_server(|headers, _body| {
            assert!(headers.starts_with("GET /v1/me/player "));
            (
                403,
                json!({"error": {"message": "Premium required"}}).to_string(),
                Vec::new(),
            )
        });
        let old_api = env::var("HERMES_SPOTIFY_API_BASE_URL").ok();
        with_env_var(
            "HERMES_SPOTIFY_API_BASE_URL",
            Some(&format!("{base_url}/v1")),
        );

        let payload = serde_json::from_str::<Value>(&handle_spotify_playback(
            &json!({"action": "get_state"}),
            &runtime_for(&home),
        ))
        .unwrap();
        assert!(
            payload["error"]
                .as_str()
                .unwrap()
                .contains("Playback control usually requires a Spotify Premium account")
        );

        join.join().unwrap();
        with_env_var("HERMES_SPOTIFY_API_BASE_URL", old_api.as_deref());
    }
}
