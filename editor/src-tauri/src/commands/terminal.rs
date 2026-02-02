//! Terminal Emulator
//!
//! This module provides PTY-based terminal emulation for the integrated terminal.
//! It supports creating multiple terminal instances, writing input, resizing,
//! and streaming output back to the frontend.

use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use tauri::{AppHandle, Emitter, State};

/// Terminal instance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalInfo {
    pub id: String,
    pub title: String,
    pub cwd: String,
    pub cols: u16,
    pub rows: u16,
}

/// Internal terminal instance
struct TerminalInstance {
    #[allow(dead_code)]
    info: TerminalInfo,
    writer: Box<dyn Write + Send>,
    #[allow(dead_code)]
    master: Box<dyn portable_pty::MasterPty + Send>,
}

/// State for managing terminal instances
pub struct TerminalState {
    terminals: Arc<Mutex<HashMap<String, TerminalInstance>>>,
    counter: Arc<Mutex<u32>>,
}

impl Default for TerminalState {
    fn default() -> Self {
        Self {
            terminals: Arc::new(Mutex::new(HashMap::new())),
            counter: Arc::new(Mutex::new(0)),
        }
    }
}

/// Create a new terminal instance
#[tauri::command]
pub async fn create_terminal(
    app: AppHandle,
    state: State<'_, TerminalState>,
    cwd: Option<String>,
    cols: Option<u16>,
    rows: Option<u16>,
) -> Result<TerminalInfo, String> {
    let pty_system = native_pty_system();

    let cols = cols.unwrap_or(80);
    let rows = rows.unwrap_or(24);

    let pair = pty_system
        .openpty(PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(|e| format!("Failed to open PTY: {}", e))?;

    // Get the default shell
    let shell = if cfg!(windows) {
        std::env::var("COMSPEC").unwrap_or_else(|_| "cmd.exe".to_string())
    } else {
        std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_string())
    };

    let mut cmd = CommandBuilder::new(&shell);

    // Set working directory
    let working_dir = cwd.clone().unwrap_or_else(|| {
        std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| "/".to_string())
    });

    cmd.cwd(&working_dir);

    // Set environment variables
    cmd.env("TERM", "xterm-256color");
    cmd.env("COLORTERM", "truecolor");

    // Spawn the child process
    let child = pair
        .slave
        .spawn_command(cmd)
        .map_err(|e| format!("Failed to spawn shell: {}", e))?;

    drop(child); // We don't need to track the child process directly

    // Generate terminal ID
    let id = {
        let mut counter = state
            .counter
            .lock()
            .map_err(|_| "Failed to lock counter")?;
        *counter += 1;
        format!("terminal-{}", *counter)
    };

    let info = TerminalInfo {
        id: id.clone(),
        title: format!("Terminal {}", id.split('-').last().unwrap_or("1")),
        cwd: working_dir,
        cols,
        rows,
    };

    // Get reader and writer
    let reader = pair
        .master
        .try_clone_reader()
        .map_err(|e| format!("Failed to clone reader: {}", e))?;

    let writer = pair
        .master
        .take_writer()
        .map_err(|e| format!("Failed to take writer: {}", e))?;

    // Store terminal instance
    {
        let mut terminals = state
            .terminals
            .lock()
            .map_err(|_| "Failed to lock terminals")?;

        terminals.insert(
            id.clone(),
            TerminalInstance {
                info: info.clone(),
                writer,
                master: pair.master,
            },
        );
    }

    // Spawn thread to read output and emit events
    let terminal_id = id.clone();
    let app_handle = app.clone();

    thread::spawn(move || {
        let mut buf_reader = BufReader::new(reader);
        let mut buffer = vec![0u8; 4096];

        loop {
            match buf_reader.fill_buf() {
                Ok(data) if data.is_empty() => break,
                Ok(data) => {
                    let len = data.len().min(buffer.len());
                    buffer[..len].copy_from_slice(&data[..len]);

                    let output = String::from_utf8_lossy(&buffer[..len]).to_string();

                    let _ = app_handle.emit(
                        "terminal-output",
                        serde_json::json!({
                            "id": terminal_id,
                            "data": output,
                        }),
                    );

                    buf_reader.consume(len);
                }
                Err(_) => break,
            }
        }

        // Emit terminal closed event
        let _ = app_handle.emit(
            "terminal-closed",
            serde_json::json!({
                "id": terminal_id,
            }),
        );
    });

    Ok(info)
}

/// Write data to a terminal
#[tauri::command]
pub async fn write_terminal(
    state: State<'_, TerminalState>,
    id: String,
    data: String,
) -> Result<(), String> {
    let mut terminals = state
        .terminals
        .lock()
        .map_err(|_| "Failed to lock terminals")?;

    let terminal = terminals
        .get_mut(&id)
        .ok_or_else(|| "Terminal not found".to_string())?;

    terminal
        .writer
        .write_all(data.as_bytes())
        .map_err(|e| format!("Failed to write to terminal: {}", e))?;

    terminal
        .writer
        .flush()
        .map_err(|e| format!("Failed to flush terminal: {}", e))?;

    Ok(())
}

/// Resize a terminal
#[tauri::command]
pub async fn resize_terminal(
    state: State<'_, TerminalState>,
    id: String,
    cols: u16,
    rows: u16,
) -> Result<(), String> {
    let terminals = state
        .terminals
        .lock()
        .map_err(|_| "Failed to lock terminals")?;

    let terminal = terminals
        .get(&id)
        .ok_or_else(|| "Terminal not found".to_string())?;

    terminal
        .master
        .resize(PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(|e| format!("Failed to resize terminal: {}", e))?;

    Ok(())
}

/// Close a terminal
#[tauri::command]
pub async fn close_terminal(state: State<'_, TerminalState>, id: String) -> Result<(), String> {
    let mut terminals = state
        .terminals
        .lock()
        .map_err(|_| "Failed to lock terminals")?;

    terminals.remove(&id);

    Ok(())
}
