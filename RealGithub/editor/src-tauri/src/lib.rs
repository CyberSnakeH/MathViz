//! MathViz Editor Library
//!
//! This module exports the core functionality of the MathViz editor backend.
//! It provides modules for file operations, terminal emulation, Git integration,
//! compiler interaction, and LSP client functionality.

pub mod commands;
pub mod lsp;

/// Re-export commonly used types
pub use commands::file::FileInfo;
pub use commands::git::GitStatus;
pub use commands::terminal::TerminalState;
