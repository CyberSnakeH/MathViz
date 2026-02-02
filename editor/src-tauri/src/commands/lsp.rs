//! LSP Commands
//!
//! Tauri commands for Language Server Protocol functionality.
//! These commands allow the frontend to interact with language servers
//! for features like autocompletion, hover info, go to definition, etc.

use crate::lsp::client::LspClient;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tauri::State;
use tokio::sync::Mutex;

/// LSP Manager state for managing multiple language server connections
pub struct LspManager {
    clients: Arc<Mutex<HashMap<String, LspClient>>>,
}

impl Default for LspManager {
    fn default() -> Self {
        Self {
            clients: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

/// LSP Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LspServerConfig {
    pub language_id: String,
    pub command: String,
    pub args: Vec<String>,
    pub root_uri: Option<String>,
}

/// Completion item returned to frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompletionItem {
    pub label: String,
    pub kind: Option<i32>,
    pub detail: Option<String>,
    pub documentation: Option<String>,
    pub insert_text: Option<String>,
    pub insert_text_format: Option<i32>,
}

/// Hover information returned to frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HoverInfo {
    pub contents: String,
    pub range: Option<Range>,
}

/// Location for go to definition
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Location {
    pub uri: String,
    pub range: Range,
}

/// Range in a document
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Range {
    pub start: Position,
    pub end: Position,
}

/// Position in a document
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Position {
    pub line: u32,
    pub character: u32,
}

/// Diagnostic information
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Diagnostic {
    pub range: Range,
    pub severity: Option<i32>,
    pub code: Option<String>,
    pub source: Option<String>,
    pub message: String,
}

/// Start an LSP server for a language
#[tauri::command]
pub async fn lsp_start(
    state: State<'_, LspManager>,
    config: LspServerConfig,
) -> Result<bool, String> {
    let mut clients = state.clients.lock().await;

    // Check if already running
    if clients.contains_key(&config.language_id) {
        return Ok(true);
    }

    // Create and start new client
    let mut client = LspClient::new();

    let args: Vec<&str> = config.args.iter().map(|s| s.as_str()).collect();
    client.start(&config.command, &args)?;

    // Initialize the server
    let _ = client.initialize(config.root_uri).await?;
    client.initialized()?;

    clients.insert(config.language_id, client);
    Ok(true)
}

/// Stop an LSP server
#[tauri::command]
pub async fn lsp_stop(
    state: State<'_, LspManager>,
    language_id: String,
) -> Result<bool, String> {
    let mut clients = state.clients.lock().await;

    if let Some(mut client) = clients.remove(&language_id) {
        let _ = client.shutdown().await;
        client.stop();
    }

    Ok(true)
}

/// Notify the LSP server that a document was opened
#[tauri::command]
pub async fn lsp_did_open(
    state: State<'_, LspManager>,
    language_id: String,
    uri: String,
    version: i32,
    text: String,
) -> Result<(), String> {
    let clients = state.clients.lock().await;

    if let Some(client) = clients.get(&language_id) {
        client.did_open(&uri, &language_id, version, &text)?;
    }

    Ok(())
}

/// Notify the LSP server that a document changed
#[tauri::command]
pub async fn lsp_did_change(
    state: State<'_, LspManager>,
    language_id: String,
    uri: String,
    version: i32,
    text: String,
) -> Result<(), String> {
    let clients = state.clients.lock().await;

    if let Some(client) = clients.get(&language_id) {
        client.did_change(&uri, version, &text)?;
    }

    Ok(())
}

/// Notify the LSP server that a document was closed
#[tauri::command]
pub async fn lsp_did_close(
    state: State<'_, LspManager>,
    language_id: String,
    uri: String,
) -> Result<(), String> {
    let clients = state.clients.lock().await;

    if let Some(client) = clients.get(&language_id) {
        client.did_close(&uri)?;
    }

    Ok(())
}

/// Request completions from the LSP server
#[tauri::command]
pub async fn lsp_completion(
    state: State<'_, LspManager>,
    language_id: String,
    uri: String,
    line: u32,
    character: u32,
) -> Result<Vec<CompletionItem>, String> {
    let clients = state.clients.lock().await;

    if let Some(client) = clients.get(&language_id) {
        let result = client.completion(&uri, line, character).await?;
        let items = parse_completion_response(result);
        return Ok(items);
    }

    Ok(vec![])
}

/// Request hover information from the LSP server
#[tauri::command]
pub async fn lsp_hover(
    state: State<'_, LspManager>,
    language_id: String,
    uri: String,
    line: u32,
    character: u32,
) -> Result<Option<HoverInfo>, String> {
    let clients = state.clients.lock().await;

    if let Some(client) = clients.get(&language_id) {
        let result = client.hover(&uri, line, character).await?;
        let hover = parse_hover_response(result);
        return Ok(hover);
    }

    Ok(None)
}

/// Request go to definition from the LSP server
#[tauri::command]
pub async fn lsp_definition(
    state: State<'_, LspManager>,
    language_id: String,
    uri: String,
    line: u32,
    character: u32,
) -> Result<Vec<Location>, String> {
    let clients = state.clients.lock().await;

    if let Some(client) = clients.get(&language_id) {
        let result = client.definition(&uri, line, character).await?;
        let locations = parse_location_response(result);
        return Ok(locations);
    }

    Ok(vec![])
}

/// Get list of active LSP servers
#[tauri::command]
pub async fn lsp_list_servers(state: State<'_, LspManager>) -> Result<Vec<String>, String> {
    let clients = state.clients.lock().await;
    Ok(clients.keys().cloned().collect())
}

// Helper functions to parse LSP responses

fn parse_completion_response(response: serde_json::Value) -> Vec<CompletionItem> {
    let mut items = Vec::new();

    // Handle both CompletionList and array of CompletionItem
    let completion_items = if let Some(list) = response.get("items") {
        list.as_array()
    } else {
        response.as_array()
    };

    if let Some(arr) = completion_items {
        for item in arr {
            if let Some(label) = item.get("label").and_then(|v| v.as_str()) {
                items.push(CompletionItem {
                    label: label.to_string(),
                    kind: item.get("kind").and_then(|v| v.as_i64()).map(|v| v as i32),
                    detail: item.get("detail").and_then(|v| v.as_str()).map(|s| s.to_string()),
                    documentation: item
                        .get("documentation")
                        .and_then(|v| {
                            if let Some(s) = v.as_str() {
                                Some(s.to_string())
                            } else if let Some(obj) = v.as_object() {
                                obj.get("value").and_then(|v| v.as_str()).map(|s| s.to_string())
                            } else {
                                None
                            }
                        }),
                    insert_text: item.get("insertText").and_then(|v| v.as_str()).map(|s| s.to_string()),
                    insert_text_format: item.get("insertTextFormat").and_then(|v| v.as_i64()).map(|v| v as i32),
                });
            }
        }
    }

    items
}

fn parse_hover_response(response: serde_json::Value) -> Option<HoverInfo> {
    let contents = response.get("contents")?;

    let content_str = if let Some(s) = contents.as_str() {
        s.to_string()
    } else if let Some(obj) = contents.as_object() {
        obj.get("value")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_default()
    } else if let Some(arr) = contents.as_array() {
        arr.iter()
            .filter_map(|v| {
                if let Some(s) = v.as_str() {
                    Some(s.to_string())
                } else if let Some(obj) = v.as_object() {
                    obj.get("value").and_then(|v| v.as_str()).map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    } else {
        return None;
    };

    if content_str.is_empty() {
        return None;
    }

    let range = response.get("range").and_then(|r| {
        Some(Range {
            start: Position {
                line: r.get("start")?.get("line")?.as_u64()? as u32,
                character: r.get("start")?.get("character")?.as_u64()? as u32,
            },
            end: Position {
                line: r.get("end")?.get("line")?.as_u64()? as u32,
                character: r.get("end")?.get("character")?.as_u64()? as u32,
            },
        })
    });

    Some(HoverInfo {
        contents: content_str,
        range,
    })
}

fn parse_location_response(response: serde_json::Value) -> Vec<Location> {
    let mut locations = Vec::new();

    // Handle both single Location and array
    let location_arr = if response.is_array() {
        response.as_array().cloned()
    } else if response.is_object() {
        Some(vec![response])
    } else {
        None
    };

    if let Some(arr) = location_arr {
        for loc in arr {
            if let (Some(uri), Some(range)) = (
                loc.get("uri").and_then(|v| v.as_str()),
                loc.get("range"),
            ) {
                if let (Some(start), Some(end)) = (range.get("start"), range.get("end")) {
                    locations.push(Location {
                        uri: uri.to_string(),
                        range: Range {
                            start: Position {
                                line: start.get("line").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                                character: start.get("character").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                            },
                            end: Position {
                                line: end.get("line").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                                character: end.get("character").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                            },
                        },
                    });
                }
            }
        }
    }

    locations
}
