"""Translate GLM's native XML tool calls to OpenAI chat messages and deltas."""
import ast
import json
import re
import uuid


def prepare_tools(messages, tools, choice, parallel=True):
    """Normalize JSON arguments for the model's bundled Jinja template.

    History the client cannot change later never fails the request (each later turn would fail the same way):
    a tool message without tool_call_id is rendered in message order (the template's own fallback), and historical
    arguments that are empty, null, not JSON or not a JSON object are rendered as {} (as vLLM does); each is logged."""
    messages = [dict(message) for message in messages]
    for message in messages:
        message["content"] = message.get("content") or ""
        if message["role"] == "developer":
            message["role"] = "system"
        if message["role"] == "tool" and not message.get("tool_call_id"):
            print("Rendering a tool message without tool_call_id in message order", flush=True)
        normalized = []
        for call in message.get("tool_calls") or []:
            call = {**call, "function": dict(call["function"])}
            arguments = call["function"].get("arguments", "{}")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (ValueError, RecursionError):
                    arguments = None
            if not isinstance(arguments, dict):
                print(f"Rendering historical arguments of {call['function'].get('name')!r} as {{}}: not a JSON object", flush=True)
                arguments = {}
            call["function"]["arguments"] = arguments
            normalized.append(call)
        if normalized:
            if message["role"] != "assistant":
                raise ValueError("Only assistant messages can contain tool_calls")
            message["tool_calls"] = normalized
    tools = tools or []
    names = [tool["function"]["name"] for tool in tools]
    if len(set(names)) != len(names):
        raise ValueError("Tool names must be unique")
    forced = None
    if isinstance(choice, dict):
        forced = (choice.get("function") or {}).get("name")
        if forced not in names:
            raise ValueError("tool_choice names a function absent from tools")
        tools = [tool for tool in tools if tool["function"]["name"] == forced]
    elif choice == "required" and not tools:
        raise ValueError("tool_choice required needs at least one tool")
    elif choice == "none":
        tools = []
    instruction = ""
    if choice == "required" or forced:
        instruction = (f"You must call the function {forced} on this turn." if forced
                       else "You must call at least one of the provided functions on this turn.")
    if tools and not parallel:
        instruction += " Call at most one function on this turn."
    if choice == "none":
        instruction = "Do not call functions on this turn. Answer directly."
    if instruction:
        messages.insert(0, {"role": "system", "content": instruction.strip()})
    return messages, tools


_FUNCTION_NAME = re.compile(r"[A-Za-z0-9_.:\-]{1,128}")


def names_strict(policy, tool_choice):
    """Whether a call to a function absent from the request's tools must fail the request.

    Strict when the operator asks for it (policy "error") or when the client constrained the call:
    a forced function or tool_choice=required is only satisfied by an advertised function."""
    return policy == "error" or tool_choice == "required" or isinstance(tool_choice, dict)


_JSON_TYPES = ("object", "array", "number", "integer", "boolean", "null")


def _loose_value(raw, kind):
    """A tool argument the model wrote that is not valid JSON: read it as a Python literal ('a', 'b' / True / None), as vLLM's GLM parser does.
    Returns (True, value) only when the literal has the schema's type; otherwise (False, None)."""
    try:
        value = ast.literal_eval(raw.strip())
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False, None
    if kind == "array" and isinstance(value, tuple):
        value = list(value)
    if kind == "boolean":
        ok = isinstance(value, bool)
    elif kind == "integer":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "number":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif kind == "null":
        ok = value is None
    elif kind == "array":
        ok = isinstance(value, list)
    else:
        ok = isinstance(value, dict)
    if ok:
        try:  # a literal can hold values JSON cannot carry (bytes, sets, complex, tuple keys)
            json.dumps(value)
        except (TypeError, ValueError, RecursionError):
            ok = False
    return (True, value) if ok else (False, None)


def parse_call(block, tools, strict_names=True, strict_args=None):
    """The GLM template uses arg_key/arg_value pairs, not a JSON call body.

    A well-formed call to a function the client did not advertise (the model recalls tools from the
    conversation history or system prompt) is handed to the client when strict_names is false: the
    client owns tool execution and answers an unknown name itself, so the model can recover.

    An argument whose schema type is not string but whose text is not valid JSON is read as a Python literal when it has the schema's type; otherwise,
    unless strict_args (default: the value of strict_names), it is passed to the client as the raw string, which validates it and lets the model retry."""
    if strict_args is None:
        strict_args = strict_names
    name, _, _ = block.partition("<arg_key>")
    name = name.strip()
    function = next((tool["function"] for tool in tools
                     if tool["function"]["name"] == name), None)
    if function is None:
        if strict_names or not _FUNCTION_NAME.fullmatch(name):
            raise ValueError(f"Model called an unadvertised function: {name}")
        print(f"Passing through a call to an unadvertised function: {name!r}", flush=True)
        function = {}
    pairs = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
    arguments = {}
    rest = block[len(block) - len(block.lstrip()):]
    rest = rest[len(name):]
    # Client schemas are not validated: a non-object "parameters"/"properties"/property schema (JSON Schema allows
    # true/false) means "no type known" instead of an AttributeError after the response has started.
    parameters = function.get("parameters")
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
    properties = properties if isinstance(properties, dict) else {}
    position = 0
    for match in pairs.finditer(rest):
        if rest[position:match.start()].strip():
            raise ValueError("Malformed model tool arguments")
        key, raw = match.groups()
        key = key.strip()
        if not key or key in arguments:
            raise ValueError("Empty or repeated model tool argument")
        schema = properties.get(key)
        schema = schema if isinstance(schema, dict) else {}
        if schema.get("type") == "string":
            value = raw
        else:
            try:
                value = json.loads(raw)
            except (ValueError, RecursionError) as error:
                kind = schema.get("type")
                if kind in _JSON_TYPES:
                    ok, loose = _loose_value(raw, kind)
                    if ok:
                        value = loose
                    elif strict_args:
                        raise ValueError(f"Invalid JSON value for tool argument {key}") from error
                    else:
                        print(f"Passing through a non-JSON value for tool argument {key!r} of {name!r} (schema type {kind})", flush=True)
                        value = raw
                else:
                    value = raw
        arguments[key] = value
        position = match.end()
    if rest[position:].strip():
        raise ValueError("Malformed model tool arguments")
    return {"id": "call_" + uuid.uuid4().hex, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}


class ToolSplitter:
    """Stream visible text; hold only an unfinished native tool-call block.

    strict_calls (default True) raises for a call block that cannot be parsed, a second call with
    parallel_tool_calls=false, and a call that generation ended inside without reaching the token limit.
    When false (the API's non-strict mode) the model's deviation is not run and does not fail the response:
    an unparseable or unterminated block is returned as ordinary text content, an extra call is dropped
    (vLLM keeps only the first), and one line is logged."""
    OPEN = "<tool_call>"
    CLOSE = "</tool_call>"

    def __init__(self, tools, parallel=True, strict_names=True, strict_args=None, strict_calls=True):
        self.tools = tools
        self.parallel = parallel
        self.strict_names = strict_names
        self.strict_args = strict_names if strict_args is None else strict_args
        self.strict_calls = strict_calls
        self.pending = ""
        self.in_call = False
        self.calls = []

    def feed(self, text, final=False, truncated=False):
        self.pending += text
        deltas = []
        while self.pending:
            if self.in_call:
                end = self.pending.find(self.CLOSE)
                if end < 0:
                    break
                block = self.pending[:end]
                self.pending = self.pending[end + len(self.CLOSE):]
                self.in_call = False
                try:
                    call = parse_call(block, self.tools, self.strict_names, self.strict_args)
                except ValueError as error:
                    if self.strict_calls:
                        raise
                    print(f"Returning an unparseable tool call as text: {error}", flush=True)
                    deltas.append({"content": self.OPEN + block + self.CLOSE})
                    continue
                if self.calls and not self.parallel:
                    if self.strict_calls:
                        raise ValueError("Model returned multiple calls with parallel_tool_calls=false")
                    print(f"Dropping an extra call to {call['function']['name']!r}: parallel_tool_calls=false", flush=True)
                    continue
                self.calls.append(call)
                deltas.append({"tool_calls": [{"index": len(self.calls) - 1, **call}]})
            else:
                start = self.pending.find(self.OPEN)
                if start >= 0:
                    if start:
                        deltas.append({"content": self.pending[:start]})
                    self.pending = self.pending[start + len(self.OPEN):]
                    self.in_call = True
                    continue
                held = 0
                if not final:
                    for count in range(1, len(self.OPEN)):
                        if self.pending.endswith(self.OPEN[:count]):
                            held = count
                visible = self.pending[:-held] if held else self.pending
                self.pending = self.pending[-held:] if held else ""
                if visible:
                    deltas.append({"content": visible})
                break
        if final and self.in_call:
            if not truncated:
                if self.strict_calls:
                    raise ValueError("Model returned an incomplete tool call")
                print("Returning an incomplete tool call as text: generation ended before </tool_call>", flush=True)
                deltas.append({"content": self.OPEN + self.pending})
            # A token limit must not execute an incomplete call or expose its XML.
            self.pending = ""
            self.in_call = False
        return deltas
