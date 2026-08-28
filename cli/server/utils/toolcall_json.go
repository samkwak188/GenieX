// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package utils

import (
	"errors"
	"strings"

	"github.com/bytedance/sonic"
	"github.com/bytedance/sonic/ast"
	"github.com/openai/openai-go/v3"
)

// jsonFormat covers models wrapping a tool call in `{"name":…, "arguments":…}`.
type jsonFormat struct{}

var errJSON = errors.New("json tool call not match")

// balancedObjectEnd returns the index just past the '}' matching the '{' at
// start, skipping braces inside string literals. -1 if never closed.
func balancedObjectEnd(s string, start int) int {
	depth := 0
	inStr, escaped := false, false
	for i := start; i < len(s); i++ {
		c := s[i]
		if inStr {
			switch {
			case escaped:
				escaped = false
			case c == '\\':
				escaped = true
			case c == '"':
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return i + 1
			}
		}
	}
	return -1
}

// scanObjects visits every '{' in s with its balanced end (-1 if never closed)
// and returns the first offset visit accepts, or -1. A reject skips the object.
func scanObjects(s string, visit func(start, end int) bool) int {
	for i := 0; i < len(s); {
		j := strings.IndexByte(s[i:], '{')
		if j < 0 {
			return -1
		}
		i += j
		end := balancedObjectEnd(s, i)
		if visit(i, end) {
			return i
		}
		if end < 0 {
			i++
		} else {
			i = end
		}
	}
	return -1
}

// parseToolCallObject decodes one JSON object into a tool call, succeeding only
// when "name" is a string and "arguments" is an object or string.
func parseToolCallObject(obj string) (openai.ChatCompletionMessageFunctionToolCallFunction, error) {
	toolCall := openai.ChatCompletionMessageFunctionToolCallFunction{}

	name, err := sonic.GetFromString(obj, "name")
	if err != nil {
		return toolCall, err
	}
	if name.TypeSafe() != ast.V_STRING {
		return toolCall, errors.New("name is not a string")
	}
	toolCall.Name, err = name.String()
	if err != nil {
		return toolCall, err
	}

	arguments, err := sonic.GetFromString(obj, "arguments")
	if err != nil {
		return toolCall, err
	}
	switch arguments.TypeSafe() {
	case ast.V_OBJECT:
		toolCall.Arguments, _ = arguments.Raw()
	case ast.V_STRING:
		toolCall.Arguments, _ = arguments.String()
	default:
		return toolCall, errors.New("unknown arguments type")
	}

	return toolCall, nil
}

func (jsonFormat) Parse(s string) (openai.ChatCompletionMessageFunctionToolCallFunction, error) {
	var toolCall openai.ChatCompletionMessageFunctionToolCallFunction
	found := scanObjects(s, func(start, end int) bool {
		if end < 0 {
			return false
		}
		var err error
		toolCall, err = parseToolCallObject(s[start:end])
		return err == nil
	})
	if found < 0 {
		return openai.ChatCompletionMessageFunctionToolCallFunction{}, errJSON
	}
	return toolCall, nil
}

func (jsonFormat) Boundary(s string) int {
	found := scanObjects(s, func(start, end int) bool {
		if end < 0 {
			return true // still open: may yet close into one
		}
		_, err := parseToolCallObject(s[start:end])
		return err == nil
	})
	if found < 0 {
		return len(s)
	}
	return found
}

// A held '{' only moves once it closes; '{' is one byte, so tail cannot be partial.
func (jsonFormat) StaysHeld(_ int, token string) bool {
	return !strings.Contains(token, "}")
}
