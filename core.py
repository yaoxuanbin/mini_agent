"""
The agent loop.

run_agent()       — single provider, all tools run locally
run_agent_mixed() — local Ollama orchestrates; routes complex tasks to a remote LLM
"""

from __future__ import annotations

from providers import LLMProvider, LLMMessage
from tools import TOOL_SCHEMAS, call_tool
from ui import Spinner

_SYSTEM_DEFAULT = (
    "You are a helpful assistant. Use the available tools when needed. "
    "When you have a complete final answer, respond in plain text without calling any tools."
)

_SYSTEM_MIXED = (
    "You are a local AI orchestrator running on the user's machine. "
    "You have access to local tools: get_current_date, calculate, get_weather, web_search. "
    "You also have ask_remote() to delegate tasks to a more capable cloud model. "
    "Use local tools for simple lookups, calculations, and factual queries. "
    "Use ask_remote() for tasks that need deep reasoning, creative writing, detailed "
    "explanations, or broad world knowledge that you are not confident about. "
    "Always assemble the final answer yourself."
)

_ASK_REMOTE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "ask_remote",
        "description": (
            "Delegate a task to the more capable remote AI model. "
            "Use this for complex reasoning, creative writing, detailed explanations, "
            "or questions that require broad world knowledge."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The specific question or subtask to send to the remote model.",
                }
            },
            "required": ["task"],
        },
    },
}


def _fmt_args(arguments: dict) -> str:
    s = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
    return (s[:80] + "…") if len(s) > 80 else s


def run_agent(
    task: str,
    provider: LLMProvider,
    tools: list[dict] | None = None,
    extra_functions: dict | None = None,
    system_prompt: str = _SYSTEM_DEFAULT,
    verbose: bool = True,
    tool_dispatcher=None,
    _spinner: Spinner | None = None,
    history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """
    Core agent loop.

    Sends the task to the provider, executes tool calls, feeds results back,
    and repeats until the model returns a plain-text final answer.

    extra_functions: optional dict of {name: callable} for dynamically injected
                     tools (e.g. ask_remote in mixed mode).
    _spinner: caller-supplied Spinner; run_agent_mixed passes its own so that
              ask_remote() can update the same spinner line.
    history: prior conversation messages to prepend (without system prompt or
             current task). Returned alongside the final answer so the caller
             can keep the conversation going across turns.

    Returns:
        (final_answer, full_messages) — full_messages is the complete message
        history including the system prompt, history, current turn, any tool
        calls/results, and the final assistant reply. The caller should slice
        off the system prompt (i.e. full_messages[1:]) when feeding it back
        as history on the next turn.
    """
    if tools is None:
        tools = TOOL_SCHEMAS

    spinner: Spinner | None = (_spinner or Spinner()) if verbose else None

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": task})

    while True:
        if spinner:
            spinner.start("Thinking…")
        try:
            response: LLMMessage = provider.complete(messages, tools)
        finally:
            if spinner:
                spinner.stop()

        messages.append(response.to_dict())

        if not response.tool_calls:
            return response.content or "", messages

        dispatch = tool_dispatcher or call_tool

        for tc in response.tool_calls:
            args_str = _fmt_args(tc.arguments)

            if spinner:
                spinner.start(f"{tc.name}({args_str})")
            try:
                if extra_functions and tc.name in extra_functions:
                    result = extra_functions[tc.name](**tc.arguments)
                else:
                    result = dispatch(tc.name, tc.arguments)
            finally:
                if spinner:
                    spinner.stop()

            if spinner:
                result_preview = str(result)
                if len(result_preview) > 100:
                    result_preview = result_preview[:100] + "…"
                spinner.println(f"  ✓ {tc.name}({args_str}) → {result_preview}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })


def run_agent_mixed(
    task: str,
    local_provider: LLMProvider,
    remote_provider: LLMProvider,
    verbose: bool = True,
    history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """
    Mixed-mode agent loop.

    The local Ollama model acts as orchestrator: it handles simple tool calls
    itself and routes complex subtasks to the remote model via ask_remote().
    """
    spinner: Spinner | None = Spinner() if verbose else None

    def ask_remote(task: str) -> str:
        task_preview = (task[:60] + "…") if len(task) > 60 else task
        if spinner:
            spinner.update(f"→ remote ({remote_provider}): {task_preview}")
        result = run_agent(task, remote_provider, verbose=False)
        if spinner:
            spinner.update("← remote: done")
        return result

    mixed_tools = TOOL_SCHEMAS + [_ASK_REMOTE_SCHEMA]

    return run_agent(
        task=task,
        provider=local_provider,
        tools=mixed_tools,
        extra_functions={"ask_remote": ask_remote},
        system_prompt=_SYSTEM_MIXED,
        verbose=verbose,
        _spinner=spinner,
        history=history,
    )


def run_agent_with_history(
    task: str,
    provider: LLMProvider,
    history: list[dict] | None = None,
    tools: list[dict] | None = None,
    extra_functions: dict | None = None,
    system_prompt: str = _SYSTEM_DEFAULT,
    verbose: bool = True,
) -> tuple[str, list[dict]]:
    """
    Convenience wrapper for conversation-style usage: feeds history in, gets
    history back (already stripped of the system prompt). Equivalent to
    calling run_agent(..., history=history) and slicing full_messages[1:].
    """
    answer, full_messages = run_agent(
        task=task,
        provider=provider,
        tools=tools,
        extra_functions=extra_functions,
        system_prompt=system_prompt,
        verbose=verbose,
        history=history,
    )
    # Drop the system prompt before handing the conversation back.
    return answer, full_messages[1:]
