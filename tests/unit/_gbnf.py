"""Test-only GBNF recogniser for the subset used by grammars/*.gbnf.

Supported: rules (`name ::= ...`), comments, literals with escapes, character
classes (ranges, negation, \\xNN, \\uNNNN, backslash escapes), alternation,
groups, `?` `*` `+` `{m}` `{m,n}`, and rule references. Recursion is not
supported (the Sair grammars are deliberately non-recursive), so each rule is
inlined and compiled to a Python regex that must match the whole input.
"""

from __future__ import annotations

import re
from pathlib import Path

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9-]*")
_ESC = {
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "\\": "\\",
    '"': '"',
    "]": "]",
    "[": "[",
    "^": "^",
    "-": "-",
    "/": "/",
}


class GbnfError(ValueError):
    pass


def _read_escape(s: str, i: int) -> tuple[str, int]:
    """s[i] is the char after a backslash. Returns (char, next index)."""
    c = s[i]
    if c == "x":
        return chr(int(s[i + 1 : i + 3], 16)), i + 3
    if c == "u":
        return chr(int(s[i + 1 : i + 5], 16)), i + 5
    if c == "U":
        return chr(int(s[i + 1 : i + 9], 16)), i + 9
    if c in _ESC:
        return _ESC[c], i + 1
    raise GbnfError(f"bad escape \\{c}")


class _Parser:
    def __init__(self, text: str) -> None:
        self.rules: dict[str, str] = {}
        self.s = ""
        self.i = 0
        self._parse_file(text)

    def _parse_file(self, text: str) -> None:
        # Strip comments, then split into rules at lines that start a new `name ::=`.
        lines = []
        for line in text.splitlines():
            in_str = False
            in_cls = False
            out = []
            j = 0
            while j < len(line):
                ch = line[j]
                if ch == "\\" and (in_str or in_cls):
                    out.append(line[j : j + 2])
                    j += 2
                    continue
                if ch == '"' and not in_cls:
                    in_str = not in_str
                elif ch == "[" and not in_str:
                    in_cls = True
                elif ch == "]" and in_cls:
                    in_cls = False
                elif ch == "#" and not in_str and not in_cls:
                    break
                out.append(ch)
                j += 1
            lines.append("".join(out))
        chunks: list[list[str]] = []
        for line in lines:
            if re.match(r"^[A-Za-z][A-Za-z0-9-]*\s*::=", line):
                chunks.append([line])
            elif line.strip():
                if not chunks:
                    raise GbnfError("text before first rule")
                chunks[-1].append(line)
        for chunk in chunks:
            body = " ".join(chunk)
            name, _, rhs = body.partition("::=")
            name = name.strip()
            if name in self.rules:
                raise GbnfError(f"duplicate rule {name}")
            self.rules[name] = rhs.strip()
        if "root" not in self.rules:
            raise GbnfError("no root rule")

    # --- regex compilation -------------------------------------------------
    def compile(self, name: str, stack: tuple[str, ...] = ()) -> str:
        if name in stack:
            raise GbnfError(f"recursive rule {name}")
        if name not in self.rules:
            raise GbnfError(f"undefined rule {name}")
        prev = (self.s, self.i)
        self.s = self.rules[name]
        self.i = 0
        out = self._alt(stack + (name,))
        self._ws()
        if self.i != len(self.s):
            raise GbnfError(f"trailing junk in rule {name}: {self.s[self.i :][:20]!r}")
        self.s, self.i = prev
        return out

    def _ws(self) -> None:
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1

    def _alt(self, stack: tuple[str, ...]) -> str:
        branches = [self._seq(stack)]
        while True:
            self._ws()
            if self.i < len(self.s) and self.s[self.i] == "|":
                self.i += 1
                branches.append(self._seq(stack))
            else:
                break
        return "(?:" + "|".join(branches) + ")"

    def _seq(self, stack: tuple[str, ...]) -> str:
        parts: list[str] = []
        while True:
            self._ws()
            if self.i >= len(self.s) or self.s[self.i] in "|)":
                break
            parts.append(self._postfix(self._atom(stack)))
        return "".join(parts)

    def _postfix(self, atom: str) -> str:
        if self.i < len(self.s):
            c = self.s[self.i]
            if c in "?*+":
                self.i += 1
                return f"(?:{atom}){c}"
            if c == "{":
                m = re.compile(r"\{(\d+)(?:(,)(\d*))?\}").match(self.s, self.i)
                if not m:
                    raise GbnfError("bad repetition")
                self.i = m.end()
                lo = m.group(1)
                hi = m.group(3) if m.group(2) else lo
                return f"(?:{atom}){{{lo},{hi or ''}}}"
        return atom

    def _atom(self, stack: tuple[str, ...]) -> str:
        c = self.s[self.i]
        if c == '"':
            self.i += 1
            buf = []
            while self.s[self.i] != '"':
                if self.s[self.i] == "\\":
                    ch, self.i = _read_escape(self.s, self.i + 1)
                else:
                    ch = self.s[self.i]
                    self.i += 1
                buf.append(re.escape(ch))
            self.i += 1
            return "".join(buf)
        if c == "[":
            return self._charclass()
        if c == "(":
            self.i += 1
            inner = self._alt(stack)
            self._ws()
            if self.s[self.i] != ")":
                raise GbnfError("missing )")
            self.i += 1
            return inner
        m = _NAME.match(self.s, self.i)
        if not m:
            raise GbnfError(f"unexpected {c!r}")
        self.i = m.end()
        return self.compile(m.group(0), stack)

    def _charclass(self) -> str:
        self.i += 1
        neg = False
        if self.s[self.i] == "^":
            neg = True
            self.i += 1
        items: list[str] = []
        while self.s[self.i] != "]":
            lo = self._class_char()
            if self.s[self.i] == "-" and self.s[self.i + 1] != "]":
                self.i += 1
                hi = self._class_char()
                items.append(f"{re.escape(lo)}-{re.escape(hi)}")
            else:
                items.append(re.escape(lo))
        self.i += 1
        if not items:
            raise GbnfError("empty class")
        return "[" + ("^" if neg else "") + "".join(items) + "]"

    def _class_char(self) -> str:
        if self.s[self.i] == "\\":
            ch, self.i = _read_escape(self.s, self.i + 1)
            return ch
        ch = self.s[self.i]
        self.i += 1
        return ch


class Grammar:
    def __init__(self, text: str) -> None:
        parser = _Parser(text)
        self.rule_names = frozenset(parser.rules)
        self.text = text
        self._re = re.compile(parser.compile("root"), re.DOTALL)

    @classmethod
    def from_file(cls, path: str | Path) -> Grammar:
        return cls(Path(path).read_text(encoding="utf-8"))

    def accepts(self, s: str) -> bool:
        return self._re.fullmatch(s) is not None
