import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2] / "glm"
sys.path.insert(0, str(ROOT))
from glm_tools import ToolSplitter, parse_call

TOOLS = [{"function": {"name": "bash", "parameters": {"properties": {
    "command": {"type": "string"}, "timeout": {"type": "integer"}}}}}]


class ToolParser(unittest.TestCase):
    def test_preserves_argument_values_and_unicode_spacing(self):
        block = "bash <arg_key>command</arg_key>\u2003<arg_value>echo '<arg_key>'\n</arg_value> <arg_key>timeout</arg_key><arg_value>5</arg_value>"
        call = parse_call(block, TOOLS)
        self.assertEqual(json.loads(call["function"]["arguments"]), {"command": "echo '<arg_key>'\n", "timeout": 5})

    def test_string_and_loose_typed_values(self):
        tools = [{"function": {"name": "f", "parameters": {"properties": {"items": {"type": "array"}}}}}]
        result = parse_call("f<arg_key>items</arg_key><arg_value>('a', 'b')</arg_value>", tools)
        self.assertEqual(json.loads(result["function"]["arguments"]), {"items": ["a", "b"]})

    def test_malformed_fragments_stay_non_executable(self):
        for rest in ["<arg_key>x", "<arg_key>x</arg_key>", "<arg_key>x</arg_key><arg_value>y", "junk", "<arg_key>x</arg_key><arg_value>1</arg_value>junk"]:
            with self.subTest(rest=rest), self.assertRaises(ValueError):
                parse_call("bash" + rest, TOOLS)

    def test_repeated_open_tags_cannot_stall_the_process(self):
        code = f"import sys; sys.path.insert(0, {str(ROOT)!r}); from glm_tools import parse_call\ntry: parse_call('bash' + '<arg_key>' * 150000, [])\nexcept ValueError: pass"
        # Non-strict name parsing ensures this exercises the arguments, not name rejection.
        code = code.replace(", [])", ", [{'function': {'name': 'bash'}}])")
        subprocess.run([sys.executable, "-c", code], check=True, timeout=3)

    def test_non_strict_stream_returns_malformed_block_as_text(self):
        text = "<tool_call>bash" + "<arg_key>" * 10000 + "</tool_call>"
        parser = ToolSplitter(TOOLS, strict_calls=False)
        result = parser.feed(text, final=True)
        self.assertEqual(result, [{"content": text}])
        self.assertEqual(parser.calls, [])

    def test_streamed_valid_call_is_unchanged(self):
        parser = ToolSplitter(TOOLS)
        text = "<tool_call>bash<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>"
        deltas = []
        for char in text: deltas.extend(parser.feed(char))
        deltas.extend(parser.feed("", final=True))
        self.assertEqual(len(parser.calls), 1)
        self.assertEqual(json.loads(parser.calls[0]["function"]["arguments"]), {"command": "pwd"})
