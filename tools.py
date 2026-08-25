"""
Tool implementations and schemas for the mini agent.

Add your own tools here:
  1. Write the Python function
  2. Add it to TOOL_FUNCTIONS
  3. Add its JSON schema to TOOL_SCHEMAS
"""

import datetime as _dt
import re
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

_MCP_SERVER = str(Path(__file__).parent / "mcp_server.py")

_WORKSPACE = Path("workspace")

# `run_command` safety configuration.
#
# FULLY OPEN MODE (user request: "完全裸奔" / "全部放开").
# - _ALLOWED_CMD_PREFIXES = None: no whitelist, any command runs.
# - _BLOCKED_SUBSTRINGS = (): no substring block either.
# - run_command passes the raw string to /bin/sh -c (shell=True).
#
# The ONLY remaining guards are:
#   * `_RUN_CMD_TIMEOUT_S` — command will be killed after N seconds.
#   * `_RUN_CMD_MAX_OUTPUT_CHARS` — output is truncated.
# That's it. Everything else is on the model's honesty.
#
# !! Implication: the model can run any shell snippet, including ones that
# !! exfiltrate credentials, install backdoors, or destroy data. Treat any
# !! prompt-injected content as live shell.
_ALLOWED_CMD_PREFIXES: tuple[str, ...] | None = None
_BLOCKED_SUBSTRINGS: tuple[str, ...] = ()

_RUN_CMD_TIMEOUT_S = 30
_RUN_CMD_MAX_OUTPUT_CHARS = 8000


def _safe_path(relative_path: str) -> Path:
    target = (_WORKSPACE / relative_path).resolve()
    try:
        target.relative_to(_WORKSPACE.resolve())
    except ValueError:
        raise ValueError(f"path '{relative_path}' escapes the workspace")
    return target


# --- Implementations ---

def get_current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    # eval with empty builtins — safe enough for a demo, not for production
    _locals = {
        "datetime": _dt,
        "date": _dt.date,
        "timedelta": _dt.timedelta,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, _locals)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str) -> str:
    # Stub — replace with OpenWeatherMap, WeatherAPI, etc.
    stub = {
        "tokyo": "18°C, partly cloudy",
        "london": "12°C, overcast",
        "new york": "22°C, sunny",
        "san francisco": "16°C, foggy",
        "berlin": "14°C, light rain",
    }
    return stub.get(city.lower(), f"20°C, clear skies")


def read_file(path: str) -> str:
    try:
        return _safe_path(path).read_text()
    except FileNotFoundError:
        return f"Error: '{path}' not found in workspace"
    except ValueError as e:
        return f"Error: {e}"


def write_file(path: str, content: str) -> str:
    try:
        p = _safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} chars to '{path}'"
    except ValueError as e:
        return f"Error: {e}"


def _import_call_mcp():
    try:
        from mcp_client import call_mcp
        return call_mcp
    except ImportError:
        raise RuntimeError("mcp package not installed — run: pip install mcp")


def mcp_to_uppercase(text: str) -> str:
    return _import_call_mcp()(_MCP_SERVER, "to_uppercase", text=text)


def mcp_count_words(text: str) -> int:
    return _import_call_mcp()(_MCP_SERVER, "count_words", text=text)


def web_search(query: str) -> str:
    # Stub — replace with Brave Search API, SerpAPI, Tavily, etc.
    return (
        f"[stub] Search results for '{query}':\n"
        f"1. Wikipedia: {query.split()[0].title()} — comprehensive overview\n"
        f"2. Recent article: '{query}' explained in 5 minutes\n"
        f"3. Official docs: {query.split()[0].lower()}.org"
    )


# --- Registry: name → callable ---
# (defined after run_command to avoid forward reference)


# --- Schemas (OpenAI function-calling format) ---

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Returns the current date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluates a mathematical expression and returns the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A Python math expression, e.g. '2 ** 10' or '847 * 0.15'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Gets current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Tokyo'"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside workspace/, e.g. 'notes.txt'"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the workspace directory, creating it if it doesn't exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside workspace/, e.g. 'output.txt'"},
                    "content": {"type": "string", "description": "Text content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "to_uppercase",
            "description": "Convert text to uppercase. Runs via an MCP server subprocess.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to convert"}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_words",
            "description": "Count the number of words in a string. Runs via an MCP server subprocess.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to count words in"}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the web for information on a topic or question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command and return its stdout+stderr. "
                "No shell — command is parsed with shlex and run via execvp. "
                "The first token must be in the safety whitelist "
                "(ls, cat, grep, python, git, curl, etc.); known-destructive "
                "patterns are blocked. Default timeout 30s. "
                "Output is truncated to 8000 chars. "
                "Use this when no dedicated tool fits (e.g. listing files, "
                "counting lines, inspecting git status, fetching a URL)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "Shell command to run, e.g. 'ls -la' or "
                            "'git status'. Do not use && or | to chain — "
                            "call run_command multiple times if needed."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Optional timeout in seconds (default 30, max 300).",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


def _shell_safe(argv: list[str]) -> tuple[bool, str]:
    """Return (ok, reason). Fully open mode — only reject empty input."""
    if not argv:
        return False, "empty command"
    if _ALLOWED_CMD_PREFIXES is not None:
        head = argv[0]
        if head not in _ALLOWED_CMD_PREFIXES:
            return False, (
                f"command '{head}' is not in the whitelist. "
                f"Allowed prefixes: {', '.join(_ALLOWED_CMD_PREFIXES)}"
            )
    full = " ".join(argv)
    low = full.lower()
    for bad in _BLOCKED_SUBSTRINGS:
        if bad in low:
            return False, f"command contains blocked pattern '{bad}'"
    return True, ""


def run_command(command: str, timeout: int | None = None) -> str:
    """
    Run a shell command and return its combined stdout+stderr.

    !! FULLY OPEN MODE — no whitelist, no substring block. The `command`
    !! string is passed verbatim to `/bin/sh -c`, so pipes, redirects,
    !! `&&` chains, env-var expansion, globbing, command substitution,
    !! and background jobs all work. Only remaining guards are a wall-clock
    !! timeout and output-length truncation.

    Output is annotated with an exit-code footer so the agent can tell
    success from failure:
      `[exit=N]` — N is the process return code (0 = success).
    For `curl` calls (the main culprit for silent 4xx/5xx), the HTTP status
    code is also surfaced: `[exit=N http=403]`. This is critical for breaking
    out of repeat-call loops where the agent re-issues the same `curl` after
    hitting a rate limit or 404 — a JSON error body looks like a successful
    answer to the model otherwise.

    Args:
        command: Shell command line (passed to `/bin/sh -c`).
        timeout: Seconds before the command is killed (default
                 `_RUN_CMD_TIMEOUT_S`). Pass a larger value for long ops.

    The command runs in the current working directory (typically the project
    root, not inside `workspace/`). Pass `cd workspace && ...` if needed.
    """
    ok, reason = _shell_safe([command])
    if not ok:
        return f"Error: {reason}"

    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout if timeout is not None else _RUN_CMD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout or _RUN_CMD_TIMEOUT_S}s"
    except Exception as e:
        return f"Error: {e}"

    out = (completed.stdout or "") + (completed.stderr or "")
    if len(out) > _RUN_CMD_MAX_OUTPUT_CHARS:
        out = (
            out[:_RUN_CMD_MAX_OUTPUT_CHARS]
            + f"\n... [truncated, total {len(out)} chars]"
        )
    rc = completed.returncode

    # Detect curl invocations so we can surface HTTP status codes. `curl -s`
    # silently exits 0 on 4xx/5xx — without this hint, the agent has no way
    # to distinguish a successful JSON response from a rate-limit error body.
    http_code = None
    if rc == 0 and re.search(r"(^|\s|;|&)curl(\s|$)", command):
        # Prefer an explicit -w '%{http_code}' if the user already asked for
        # one; otherwise re-run once silently with -o /dev/null to extract it.
        # We only do this on success to keep behaviour predictable.
        m = re.search(r"-w\s+['\"]?[^'\"]*%\{http_code\}[^'\"]*['\"]?", command)
        if m:
            # The user's own -w template already produced http_code in stdout.
            # Don't re-invoke; just look for a 3-digit code at end of stdout.
            tail = (completed.stdout or "").strip().splitlines()
            if tail and tail[-1].strip().isdigit() and len(tail[-1].strip()) == 3:
                http_code = tail[-1].strip()
        else:
            try:
                probe = subprocess.run(
                    f"{command} -o /dev/null -w '%{{http_code}}'",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                code = (probe.stdout or "").strip()
                if code.isdigit():
                    http_code = code
            except Exception:
                pass

    # Common 4xx/5xx markers in body — flag even when we couldn't get http_code.
    error_hint = None
    if rc == 0:
        body_low = out.lower()
        if "rate limit exceeded" in body_low:
            error_hint = "github_api_rate_limit"
        elif "not found" in body_low and '"message"' in body_low:
            error_hint = "not_found"
        elif http_code and http_code.startswith(("4", "5")):
            error_hint = f"http_{http_code}"

    parts = [f"exit={rc}"]
    if http_code:
        parts.append(f"http={http_code}")
    if error_hint:
        parts.append(f"hint={error_hint}")
    footer = "[" + " ".join(parts) + "]"

    if not out:
        return f"(no output) {footer}"
    return f"{out.rstrip()}\n{footer}"


# --- Registry: name → callable ---
# (defined after run_command to avoid forward reference)

TOOL_FUNCTIONS: dict = {
    "get_current_date": get_current_date,
    "calculate": calculate,
    "get_weather": get_weather,
    "web_search": web_search,
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
    "to_uppercase": mcp_to_uppercase,
    "count_words": mcp_count_words,
}


def call_tool(name: str, arguments: dict) -> str:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"Unknown tool: '{name}'"
    try:
        return str(fn(**arguments))
    except Exception as e:
        return f"Tool '{name}' raised an error: {e}"
