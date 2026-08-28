// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import "testing"

// ParseToolCalls must route to whichever format carries the call.
func TestParseToolCalls(t *testing.T) {
	tests := []struct {
		name      string
		resp      string
		wantName  string
		wantArgs  string
		wantError bool
	}{
		{
			name:     "json",
			resp:     `{"name": "get_weather", "arguments": {"city": "Beijing"}}`,
			wantName: "get_weather",
			wantArgs: `{"city": "Beijing"}`,
		},
		{
			name:     "gemma4",
			resp:     `<|tool_call>call:get_weather{city:<|"|>Beijing<|"|>}<tool_call|>`,
			wantName: "get_weather",
			wantArgs: `{"city":"Beijing"}`,
		},
		{
			name:      "no format matches",
			resp:      "just some plain text without any call",
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseToolCalls(tt.resp)
			if tt.wantError {
				if err == nil {
					t.Fatalf("expected error, got tool call %+v", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got.Name != tt.wantName {
				t.Errorf("name = %q, want %q", got.Name, tt.wantName)
			}
			if got.Arguments != tt.wantArgs {
				t.Errorf("arguments = %q, want %q", got.Arguments, tt.wantArgs)
			}
		})
	}
}

// toolCallBoundary must report the earliest boundary any format sees, so the
// per-format cases live in the format tests and these cover the aggregation.
func TestToolCallBoundary(t *testing.T) {
	tests := []struct {
		name string
		s    string
		want int // -1 means len(s): nothing held
	}{
		{name: "empty", s: "", want: -1},
		{name: "prose only", s: "Sure, here is the answer.", want: -1},
		{name: "json alone", s: `hi {"name": "f", "arguments": {}}`, want: 3},
		{name: "gemma4 alone", s: "sure <|tool_call>call:f{}", want: 5},
		{name: "json earlier than gemma4", s: `x {"name": "g", "arguments": {}} y <|tool_call>call:f{}`, want: 2},
		{name: "gemma4 earlier than json", s: `x <|tool_call>call:f{} y {"name": "g", "arguments": {}}`, want: 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			want := tt.want
			if want < 0 {
				want = len(tt.s)
			}
			if got := toolCallBoundary(tt.s); got != want {
				t.Errorf("toolCallBoundary(%q) = %d, want %d", tt.s, got, want)
			}
		})
	}
}

// Feeding rune by rune must emit every byte once and leave a parsable tail.
func TestToolCallScanner(t *testing.T) {
	responses := []string{
		"just prose, no tool call at all",
		`prose then {"name": "f", "arguments": {"a": 1}} tail`,
		`{"name": "f", "arguments": {"a": 1}}`,
		`{"foo": 1} noise {"name": "f", "arguments": {"a": 1}}`,
		`talk <|tool_call>call:f{a:<|"|>b<|"|>}<tool_call|>`,
		`a } b { c unbalanced`,
	}

	for _, resp := range responses {
		t.Run(resp, func(t *testing.T) {
			var scanner ToolCallScanner
			var emitted string
			for _, r := range resp {
				emitted += scanner.Push(string(r))
			}
			tail := scanner.Tail()
			if emitted+tail != resp {
				t.Fatalf("emitted+tail = %q, want %q", emitted+tail, resp)
			}

			wantCall, wantErr := ParseToolCalls(resp)
			gotCall, gotErr := ParseToolCalls(tail)
			if (wantErr == nil) != (gotErr == nil) {
				t.Fatalf("tail parse error = %v, whole parse error = %v", gotErr, wantErr)
			}
			if gotCall.Name != wantCall.Name || gotCall.Arguments != wantCall.Arguments {
				t.Errorf("tail parsed %+v, whole parsed %+v", gotCall, wantCall)
			}
		})
	}
}

// Any split of a response must yield the same emitted text and tail. This is
// what guards StaysHeld: a format skipping a rescan it should not stalls here.
func TestToolCallScannerChunkSizes(t *testing.T) {
	responses := []string{
		`prose then {"name": "f", "arguments": {"a": {"b": 1}}} tail`,
		`talk <|tool_call>call:f{a:<|"|>b<|"|>}<tool_call|>`,
		`{"foo": 1} noise {"name": "f", "arguments": {}}`,
		`a { b } c { d } e`,
		// A '<' that does not open a marker must not stall the stream.
		`prose <think>more prose, no closing brace anywhere`,
		`prose <think>more prose, then a } brace`,
		`a <b <c <|too <|tool_ nothing here opens a marker`,
		`ends on a bare marker prefix <|tool`,
	}

	for _, resp := range responses {
		for _, size := range []int{1, 2, 3, 5, 17, len(resp)} {
			var scanner ToolCallScanner
			var emitted string
			for i := 0; i < len(resp); i += size {
				end := min(i+size, len(resp))
				emitted += scanner.Push(resp[i:end])
			}
			tail := scanner.Tail()
			if emitted+tail != resp {
				t.Errorf("size %d: emitted+tail = %q, want %q", size, emitted+tail, resp)
			}
			// The tail must start where the whole-response boundary says it does.
			if want := toolCallBoundary(resp); len(emitted) != want {
				t.Errorf("size %d: emitted %d bytes, boundary says %d", size, len(emitted), want)
			}
		}
	}
}
