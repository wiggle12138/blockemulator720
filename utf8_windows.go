//go:build windows
// +build windows

/*
Package main provides UTF-8 encoding support for BlockEmulator on Windows.

This file contains Windows-specific UTF-8 initialization code to ensure
proper character encoding handling across the application.
*/
package main

import (
	"os"
	"syscall"
	"unsafe"
)

// Windows API constants for UTF-8 console setup
const (
	CP_UTF8                            = 65001
	ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
	STD_OUTPUT_HANDLE                  = ^uintptr(10) + 1
	STD_ERROR_HANDLE                   = ^uintptr(11) + 1
)

var (
	kernel32               = syscall.MustLoadDLL("kernel32.dll")
	procSetConsoleCP       = kernel32.MustFindProc("SetConsoleCP")
	procSetConsoleOutputCP = kernel32.MustFindProc("SetConsoleOutputCP")
	procGetStdHandle       = kernel32.MustFindProc("GetStdHandle")
	procGetConsoleMode     = kernel32.MustFindProc("GetConsoleMode")
	procSetConsoleMode     = kernel32.MustFindProc("SetConsoleMode")
)

// initUTF8Console initializes the Windows console for UTF-8 support
func initUTF8Console() {
	// Set console input and output code page to UTF-8
	procSetConsoleCP.Call(uintptr(CP_UTF8))
	procSetConsoleOutputCP.Call(uintptr(CP_UTF8))

	// Enable virtual terminal processing for better UTF-8 support
	enableVirtualTerminalProcessing(STD_OUTPUT_HANDLE)
	enableVirtualTerminalProcessing(STD_ERROR_HANDLE)

	// Set environment variables for UTF-8
	os.Setenv("CHCP", "65001")
	os.Setenv("PYTHONIOENCODING", "utf-8")
	os.Setenv("LANG", "en_US.UTF-8")
	os.Setenv("LC_ALL", "en_US.UTF-8")
}

// enableVirtualTerminalProcessing enables virtual terminal processing for the given handle
func enableVirtualTerminalProcessing(handle uintptr) {
	h, _, _ := procGetStdHandle.Call(handle)
	if h == uintptr(syscall.InvalidHandle) {
		return
	}

	var mode uint32
	procGetConsoleMode.Call(h, uintptr(unsafe.Pointer(&mode)))
	mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING
	procSetConsoleMode.Call(h, uintptr(mode))
}

// init function that runs before main() to set up UTF-8 console
func init() {
	initUTF8Console()
}
