//! MathViz Editor - Main entry point
//!
//! This is the Tauri application entry point for the MathViz code editor.

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod commands;
mod lsp;

use commands::{compiler, file, git, lsp as lsp_commands, terminal};

fn main() {
    env_logger::init();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .manage(terminal::TerminalState::default())
        .manage(lsp_commands::LspManager::default())
        .invoke_handler(tauri::generate_handler![
            // File operations
            file::read_file,
            file::write_file,
            file::read_directory,
            file::create_file,
            file::create_directory,
            file::delete_path,
            file::rename_path,
            file::copy_path,
            file::get_file_info,
            file::watch_directory,
            file::unwatch_directory,
            file::search_files,
            // Terminal operations
            terminal::create_terminal,
            terminal::write_terminal,
            terminal::resize_terminal,
            terminal::close_terminal,
            // Compiler operations
            compiler::compile_mathviz,
            compiler::run_mathviz,
            compiler::check_syntax,
            compiler::get_completions,
            compiler::format_code,
            compiler::check_mathviz_installation,
            // Git operations
            git::git_status,
            git::git_diff,
            git::git_stage,
            git::git_unstage,
            git::git_commit,
            git::git_push,
            git::git_pull,
            git::git_branch_list,
            git::git_checkout,
            git::git_create_branch,
            git::git_log,
            git::git_init,
            // LSP operations
            lsp_commands::lsp_start,
            lsp_commands::lsp_stop,
            lsp_commands::lsp_did_open,
            lsp_commands::lsp_did_change,
            lsp_commands::lsp_did_close,
            lsp_commands::lsp_completion,
            lsp_commands::lsp_hover,
            lsp_commands::lsp_definition,
            lsp_commands::lsp_list_servers,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
