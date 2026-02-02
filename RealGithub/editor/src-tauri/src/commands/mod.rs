//! Tauri Commands Module
//!
//! This module contains all the Tauri command handlers organized by functionality:
//! - `file`: File system operations (read, write, create, delete, watch)
//! - `terminal`: Terminal emulator with PTY support
//! - `compiler`: MathViz compiler integration
//! - `git`: Git version control operations
//! - `lsp`: Language Server Protocol client operations

pub mod compiler;
pub mod file;
pub mod git;
pub mod lsp;
pub mod terminal;
