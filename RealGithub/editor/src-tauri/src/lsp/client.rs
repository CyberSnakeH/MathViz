//! LSP Client Implementation
//!
//! This module implements the Language Server Protocol client for communicating
//! with the MathViz language server. It handles initialization, document
//! synchronization, and request/response handling.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// LSP Client for MathViz language server
pub struct LspClient {
    process: Option<Child>,
    request_id: AtomicU64,
    #[allow(dead_code)]
    pending_requests: Arc<Mutex<HashMap<u64, tokio::sync::oneshot::Sender<serde_json::Value>>>>,
    writer: Option<Arc<Mutex<Box<dyn Write + Send>>>>,
}

/// LSP Message header (used for parsing LSP protocol)
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LspHeader {
    content_length: usize,
}

/// LSP Request (used internally for serialization)
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspRequest {
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    pub params: Option<serde_json::Value>,
}

/// LSP Response (used for deserializing server responses)
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspResponse {
    pub jsonrpc: String,
    pub id: Option<u64>,
    pub result: Option<serde_json::Value>,
    pub error: Option<LspError>,
}

/// LSP Notification (used internally for serialization)
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspNotification {
    pub jsonrpc: String,
    pub method: String,
    pub params: Option<serde_json::Value>,
}

/// LSP Error
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspError {
    pub code: i32,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

/// Initialize parameters
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    pub process_id: Option<u32>,
    pub root_uri: Option<String>,
    pub capabilities: ClientCapabilities,
}

/// Client capabilities
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ClientCapabilities {
    pub text_document: Option<TextDocumentClientCapabilities>,
    pub workspace: Option<WorkspaceClientCapabilities>,
}

/// Text document capabilities
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct TextDocumentClientCapabilities {
    pub synchronization: Option<SynchronizationCapabilities>,
    pub completion: Option<CompletionCapabilities>,
    pub hover: Option<HoverCapabilities>,
    pub definition: Option<DefinitionCapabilities>,
    pub references: Option<ReferencesCapabilities>,
    pub document_highlight: Option<DocumentHighlightCapabilities>,
    pub document_symbol: Option<DocumentSymbolCapabilities>,
    pub formatting: Option<FormattingCapabilities>,
    pub publish_diagnostics: Option<PublishDiagnosticsCapabilities>,
}

/// Synchronization capabilities
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct SynchronizationCapabilities {
    pub dynamic_registration: Option<bool>,
    pub will_save: Option<bool>,
    pub will_save_wait_until: Option<bool>,
    pub did_save: Option<bool>,
}

/// Completion capabilities
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CompletionCapabilities {
    pub dynamic_registration: Option<bool>,
    pub completion_item: Option<CompletionItemCapabilities>,
}

/// Completion item capabilities
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct CompletionItemCapabilities {
    pub snippet_support: Option<bool>,
    pub commit_characters_support: Option<bool>,
    pub documentation_format: Option<Vec<String>>,
}

/// Hover capabilities
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct HoverCapabilities {
    pub dynamic_registration: Option<bool>,
    pub content_format: Option<Vec<String>>,
}

/// Definition capabilities
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct DefinitionCapabilities {
    pub dynamic_registration: Option<bool>,
}

/// References capabilities
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ReferencesCapabilities {
    pub dynamic_registration: Option<bool>,
}

/// Document highlight capabilities
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct DocumentHighlightCapabilities {
    pub dynamic_registration: Option<bool>,
}

/// Document symbol capabilities
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct DocumentSymbolCapabilities {
    pub dynamic_registration: Option<bool>,
    pub hierarchical_document_symbol_support: Option<bool>,
}

/// Formatting capabilities
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct FormattingCapabilities {
    pub dynamic_registration: Option<bool>,
}

/// Publish diagnostics capabilities
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct PublishDiagnosticsCapabilities {
    pub related_information: Option<bool>,
    pub tag_support: Option<DiagnosticTagSupport>,
}

/// Diagnostic tag support
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct DiagnosticTagSupport {
    pub value_set: Option<Vec<i32>>,
}

/// Workspace capabilities
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceClientCapabilities {
    pub apply_edit: Option<bool>,
    pub workspace_edit: Option<WorkspaceEditCapabilities>,
    pub did_change_configuration: Option<DidChangeConfigurationCapabilities>,
    pub did_change_watched_files: Option<DidChangeWatchedFilesCapabilities>,
    pub symbol: Option<WorkspaceSymbolCapabilities>,
    pub workspace_folders: Option<bool>,
}

/// Workspace edit capabilities
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceEditCapabilities {
    pub document_changes: Option<bool>,
}

/// Did change configuration capabilities
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct DidChangeConfigurationCapabilities {
    pub dynamic_registration: Option<bool>,
}

/// Did change watched files capabilities
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct DidChangeWatchedFilesCapabilities {
    pub dynamic_registration: Option<bool>,
}

/// Workspace symbol capabilities
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct WorkspaceSymbolCapabilities {
    pub dynamic_registration: Option<bool>,
}

impl LspClient {
    /// Create a new LSP client
    pub fn new() -> Self {
        Self {
            process: None,
            request_id: AtomicU64::new(1),
            pending_requests: Arc::new(Mutex::new(HashMap::new())),
            writer: None,
        }
    }

    /// Start the language server
    pub fn start(&mut self, command: &str, args: &[&str]) -> Result<(), String> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start LSP server: {}", e))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "Failed to capture stdin".to_string())?;

        self.writer = Some(Arc::new(Mutex::new(Box::new(stdin) as Box<dyn Write + Send>)));
        self.process = Some(child);

        Ok(())
    }

    /// Send a request to the language server
    pub async fn request(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, String> {
        let id = self.request_id.fetch_add(1, Ordering::SeqCst);

        let request = LspRequest {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.to_string(),
            params,
        };

        let content = serde_json::to_string(&request)
            .map_err(|e| format!("Failed to serialize request: {}", e))?;

        let message = format!("Content-Length: {}\r\n\r\n{}", content.len(), content);

        if let Some(writer) = &self.writer {
            let mut writer = writer.lock().map_err(|_| "Failed to lock writer")?;
            writer
                .write_all(message.as_bytes())
                .map_err(|e| format!("Failed to write request: {}", e))?;
            writer
                .flush()
                .map_err(|e| format!("Failed to flush: {}", e))?;
        }

        // For now, return a placeholder
        // In a full implementation, we would set up a receiver channel
        Ok(serde_json::json!({}))
    }

    /// Send a notification to the language server
    pub fn notify(&self, method: &str, params: Option<serde_json::Value>) -> Result<(), String> {
        let notification = LspNotification {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params,
        };

        let content = serde_json::to_string(&notification)
            .map_err(|e| format!("Failed to serialize notification: {}", e))?;

        let message = format!("Content-Length: {}\r\n\r\n{}", content.len(), content);

        if let Some(writer) = &self.writer {
            let mut writer = writer.lock().map_err(|_| "Failed to lock writer")?;
            writer
                .write_all(message.as_bytes())
                .map_err(|e| format!("Failed to write notification: {}", e))?;
            writer
                .flush()
                .map_err(|e| format!("Failed to flush: {}", e))?;
        }

        Ok(())
    }

    /// Initialize the language server
    pub async fn initialize(&self, root_uri: Option<String>) -> Result<serde_json::Value, String> {
        let params = InitializeParams {
            process_id: Some(std::process::id()),
            root_uri,
            capabilities: ClientCapabilities {
                text_document: Some(TextDocumentClientCapabilities {
                    synchronization: Some(SynchronizationCapabilities {
                        dynamic_registration: Some(true),
                        will_save: Some(true),
                        will_save_wait_until: Some(true),
                        did_save: Some(true),
                    }),
                    completion: Some(CompletionCapabilities {
                        dynamic_registration: Some(true),
                        completion_item: Some(CompletionItemCapabilities {
                            snippet_support: Some(true),
                            commit_characters_support: Some(true),
                            documentation_format: Some(vec![
                                "markdown".to_string(),
                                "plaintext".to_string(),
                            ]),
                        }),
                    }),
                    hover: Some(HoverCapabilities {
                        dynamic_registration: Some(true),
                        content_format: Some(vec![
                            "markdown".to_string(),
                            "plaintext".to_string(),
                        ]),
                    }),
                    definition: Some(DefinitionCapabilities {
                        dynamic_registration: Some(true),
                    }),
                    references: Some(ReferencesCapabilities {
                        dynamic_registration: Some(true),
                    }),
                    document_highlight: Some(DocumentHighlightCapabilities {
                        dynamic_registration: Some(true),
                    }),
                    document_symbol: Some(DocumentSymbolCapabilities {
                        dynamic_registration: Some(true),
                        hierarchical_document_symbol_support: Some(true),
                    }),
                    formatting: Some(FormattingCapabilities {
                        dynamic_registration: Some(true),
                    }),
                    publish_diagnostics: Some(PublishDiagnosticsCapabilities {
                        related_information: Some(true),
                        tag_support: Some(DiagnosticTagSupport {
                            value_set: Some(vec![1, 2]),
                        }),
                    }),
                }),
                workspace: Some(WorkspaceClientCapabilities {
                    apply_edit: Some(true),
                    workspace_edit: Some(WorkspaceEditCapabilities {
                        document_changes: Some(true),
                    }),
                    did_change_configuration: Some(DidChangeConfigurationCapabilities {
                        dynamic_registration: Some(true),
                    }),
                    did_change_watched_files: Some(DidChangeWatchedFilesCapabilities {
                        dynamic_registration: Some(true),
                    }),
                    symbol: Some(WorkspaceSymbolCapabilities {
                        dynamic_registration: Some(true),
                    }),
                    workspace_folders: Some(true),
                }),
            },
        };

        self.request("initialize", Some(serde_json::to_value(params).unwrap()))
            .await
    }

    /// Notify that initialization is complete
    pub fn initialized(&self) -> Result<(), String> {
        self.notify("initialized", Some(serde_json::json!({})))
    }

    /// Open a document
    pub fn did_open(&self, uri: &str, language_id: &str, version: i32, text: &str) -> Result<(), String> {
        self.notify(
            "textDocument/didOpen",
            Some(serde_json::json!({
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": version,
                    "text": text
                }
            })),
        )
    }

    /// Notify document change
    pub fn did_change(&self, uri: &str, version: i32, text: &str) -> Result<(), String> {
        self.notify(
            "textDocument/didChange",
            Some(serde_json::json!({
                "textDocument": {
                    "uri": uri,
                    "version": version
                },
                "contentChanges": [
                    {
                        "text": text
                    }
                ]
            })),
        )
    }

    /// Close a document
    pub fn did_close(&self, uri: &str) -> Result<(), String> {
        self.notify(
            "textDocument/didClose",
            Some(serde_json::json!({
                "textDocument": {
                    "uri": uri
                }
            })),
        )
    }

    /// Request completion
    pub async fn completion(
        &self,
        uri: &str,
        line: u32,
        character: u32,
    ) -> Result<serde_json::Value, String> {
        self.request(
            "textDocument/completion",
            Some(serde_json::json!({
                "textDocument": {
                    "uri": uri
                },
                "position": {
                    "line": line,
                    "character": character
                }
            })),
        )
        .await
    }

    /// Request hover information
    pub async fn hover(&self, uri: &str, line: u32, character: u32) -> Result<serde_json::Value, String> {
        self.request(
            "textDocument/hover",
            Some(serde_json::json!({
                "textDocument": {
                    "uri": uri
                },
                "position": {
                    "line": line,
                    "character": character
                }
            })),
        )
        .await
    }

    /// Request definition
    pub async fn definition(
        &self,
        uri: &str,
        line: u32,
        character: u32,
    ) -> Result<serde_json::Value, String> {
        self.request(
            "textDocument/definition",
            Some(serde_json::json!({
                "textDocument": {
                    "uri": uri
                },
                "position": {
                    "line": line,
                    "character": character
                }
            })),
        )
        .await
    }

    /// Shutdown the language server
    pub async fn shutdown(&self) -> Result<(), String> {
        self.request("shutdown", None).await?;
        self.notify("exit", None)?;
        Ok(())
    }

    /// Stop the language server process
    pub fn stop(&mut self) {
        if let Some(mut process) = self.process.take() {
            let _ = process.kill();
        }
    }
}

impl Default for LspClient {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for LspClient {
    fn drop(&mut self) {
        self.stop();
    }
}
