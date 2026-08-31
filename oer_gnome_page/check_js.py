#!/usr/bin/env python
"""Catch broken JavaScript in template.html without needing node.

The failure it exists for: an editing step turning a \\n escape into a real
newline, which leaves a string literal unterminated. That is a hard syntax
error and it takes the whole page down silently -- the browser just stops.

Handles nested template literals (`a ${ `b ${c}` } d`), regex literals,
and comments, which this file uses heavily.

    python check_js.py [file.html]
"""
import re
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "template.html")
html = path.read_text(encoding="utf-8")

scripts = re.findall(r"<script>\n(.*?)\n</script>", html, re.S)
if not scripts:
    sys.exit("no inline <script> found in %s" % path)
if len(scripts) > 1:
    # Only one block is scanned. Silently skipping the others would report a
    # [PASS] that means nothing for them, so refuse instead.
    sys.exit("%s has %d inline <script> blocks; this checker scans one. "
             "Merge them, or extend check_js, rather than trust a partial "
             "[PASS]." % (path, len(scripts)))
src = scripts[0]
offset = html[: html.index(src)].count("\n")

# '/' starts a regex rather than a division when the previous significant
# character cannot end an expression.
REGEX_OK_AFTER = set("(,=:[!&|?{};+-*%~^") | {""}
# ...and after a keyword. `return /re/.test(x)` is valid JS; treating that '/'
# as division makes the closing '/' look like the START of a regex, and the
# scan then runs to EOF and reports a bogus unterminated literal.
REGEX_OK_AFTER_WORD = {"return", "typeof", "instanceof", "in", "of", "new",
                       "delete", "void", "throw", "case", "do", "else",
                       "yield", "await"}


def word_before(s, idx):
    """The identifier ending just before idx, skipping whitespace."""
    j = idx - 1
    while j >= 0 and s[j] in " \t\n\r":
        j -= 1
    end = j + 1
    while j >= 0 and (s[j].isalnum() or s[j] in "_$"):
        j -= 1
    return s[j + 1:end]

errors = []
# stack of frames: ("code", brace_depth_at_entry) or ("tpl", None)
stack = [["code", 0]]
brace = 0
paren = 0
brack = 0
opens = []          # (char, line) for every bracket still open
CLOSES = {"}": "{", ")": "(", "]": "["}
prev_sig = ""
line = 1
i = 0
n = len(src)


def fail(ln, msg):
    errors.append((ln, msg))


while i < n:
    ch = src[i]
    nxt = src[i + 1] if i + 1 < n else ""
    top = stack[-1][0]

    if ch == "\n":
        line += 1
        i += 1
        continue

    # ---------------- inside a template literal ----------------
    if top == "tpl":
        if ch == "\\":
            i += 2
            continue
        if ch == "`":
            stack.pop()
            prev_sig = "`"
            i += 1
            continue
        if ch == "$" and nxt == "{":
            stack.append(["code", brace])
            # this '{' never reaches the bracket branch below, so record it
            # here or its closer reads as a stray '}'
            opens.append(("{", line))
            brace += 1
            prev_sig = "{"
            i += 2
            continue
        i += 1
        continue

    # ---------------- inside code ----------------
    if ch in " \t":
        i += 1
        continue

    if ch == "/" and nxt == "/":
        j = src.find("\n", i)
        i = j if j != -1 else n
        continue

    if ch == "/" and nxt == "*":
        j = src.find("*/", i + 2)
        if j == -1:
            fail(line, "unterminated block comment")
            break
        line += src.count("\n", i, j)
        i = j + 2
        continue

    if ch in "'\"":
        quote, start_line = ch, line
        i += 1
        while i < n:
            c = src[i]
            if c == "\\":
                i += 2
                continue
            if c == "\n":
                fail(start_line, "unterminated %s string" % quote)
                break
            if c == quote:
                i += 1
                break
            i += 1
        else:
            fail(start_line, "unterminated %s string at EOF" % quote)
        prev_sig = quote
        continue

    if ch == "`":
        stack.append(["tpl", None])
        i += 1
        continue

    if ch == "/" and (prev_sig in REGEX_OK_AFTER
                      or word_before(src, i) in REGEX_OK_AFTER_WORD):
        start_line = line
        i += 1
        closed = False
        while i < n:
            c = src[i]
            if c == "\\":
                i += 2
                continue
            if c == "[":                      # char class
                j = src.find("]", i)
                i = j + 1 if j != -1 else i + 1
                continue
            if c == "\n":
                break
            if c == "/":
                i += 1
                closed = True
                break
            i += 1
        if not closed:
            fail(start_line, "unterminated regex literal")
        prev_sig = "/"
        continue

    if ch in "{([":
        opens.append((ch, line))
        brace += ch == "{"
        paren += ch == "("
        brack += ch == "["
    elif ch in "})]":
        # Order- and type-aware. The net counters below cannot see a stray
        # closer paired with a missing one, nor `{ ... )` -- both cancel.
        if not opens:
            fail(line, "stray '%s' - nothing is open here" % ch)
        elif opens[-1][0] != CLOSES[ch]:
            o, ol = opens.pop()
            fail(line, "'%s' opened at line %d is closed by '%s'"
                       % (o, ol + offset, ch))
        else:
            opens.pop()
        if ch == "}":
            brace -= 1
            # closing the ${ ... } of a template literal?
            if len(stack) > 1 and stack[-1][0] == "code" and brace == stack[-1][1]:
                stack.pop()
                prev_sig = "}"
                i += 1
                continue
        elif ch == ")":
            paren -= 1
        else:
            brack -= 1

    prev_sig = ch
    i += 1

if len(stack) > 1:
    fail(0, "unclosed template literal or ${ } block (%d frames left)" % (len(stack) - 1))
for o, ol in opens:
    fail(ol, "'%s' opened here is never closed" % o)

print("%s  (%d lines of script)" % (path.name, src.count("\n") + 1))
if errors:
    for ln, msg in errors[:25]:
        print("  [FAIL] %-11s %s" % ("line %d" % (ln + offset) if ln else "file", msg))
    if len(errors) > 25:
        print("  ... and %d more" % (len(errors) - 25))
    print("%d problem(s)" % len(errors))
    sys.exit(1)
print("  [PASS] no unterminated literals, brackets balanced")
