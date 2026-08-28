// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import (
	"errors"
	"log/slog"
	"strings"

	"github.com/openai/openai-go/v3"
)

// toolCallFormat is one model family's tool-call syntax, one per toolcall_*.go.
type toolCallFormat interface {
	Parse(s string) (openai.ChatCompletionMessageFunctionToolCallFunction, error)

	// Boundary returns the earliest offset in s that could start a tool call,
	// len(s) if none can. An opener cut off by the end of s counts.
	Boundary(s string) int

	// StaysHeld reports whether token leaves a boundary held behind tail bytes
	// alone. False only forces a rescan, so err toward false.
	StaysHeld(tail int, token string) bool
}

var toolCallFormats = []toolCallFormat{gemma4Format{}, jsonFormat{}}

// ParseToolCalls returns the first tool call in resp, whatever its format.
func ParseToolCalls(resp string) (openai.ChatCompletionMessageFunctionToolCallFunction, error) {
	for _, f := range toolCallFormats {
		if toolCall, err := f.Parse(resp); err == nil {
			slog.Debug("Parsed tool call", "tool_call", toolCall)
			return toolCall, nil
		}
	}
	return openai.ChatCompletionMessageFunctionToolCallFunction{}, errors.New("tool call not match")
}

// toolCallBoundary is the earliest offset any format could start at.
func toolCallBoundary(s string) int {
	n := len(s)
	for _, f := range toolCallFormats {
		if b := f.Boundary(s); b < n {
			n = b
		}
	}
	return n
}

// ToolCallScanner splits a stream into text safe to emit and a held-back tail.
type ToolCallScanner struct {
	buf  strings.Builder
	held int // held tail length, 0 when not holding
}

// Push appends token and returns the prefix now safe to emit.
func (s *ToolCallScanner) Push(token string) string {
	s.buf.WriteString(token)
	if s.held > 0 && staysHeld(s.held, token) {
		return ""
	}

	all := s.buf.String()
	n := toolCallBoundary(all)
	s.held = len(all) - n
	if n == 0 {
		return ""
	}
	// String() shares the array Reset() drops, so all[n:] stays valid.
	s.buf.Reset()
	s.buf.WriteString(all[n:])
	return all[:n]
}

// Tail returns the text held back and never emitted.
func (s *ToolCallScanner) Tail() string { return s.buf.String() }

func staysHeld(tail int, token string) bool {
	for _, f := range toolCallFormats {
		if !f.StaysHeld(tail, token) {
			return false
		}
	}
	return true
}
