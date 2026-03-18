"""Registry for Hannah tool modules."""

from __future__ import annotations

import json
import importlib
import inspect
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from hannah.utils.console import Console, Table

console = Console()

_REQUIRED_PARAM_KINDS = {
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
    inspect.Parameter.KEYWORD_ONLY,
}


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    module_name: str
    module: ModuleType
    parameters: dict[str, Any]
    run_fn: Callable[..., Any]
    signature: inspect.Signature


def normalize_tool_args(
    tool_name: str,
    args: dict[str, Any],
    *,
    parameters: dict[str, Any] | None = None,
    signature: inspect.Signature | None = None,
) -> dict[str, Any]:
    """Normalize and validate tool arguments against schema and callable shape."""
    if not isinstance(args, dict):
        raise ValueError(f"invalid arguments for tool '{tool_name}': expected an object payload")

    schema = parameters if isinstance(parameters, dict) else {"type": "object", "properties": {}}
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required_keys = _required_argument_names(schema, signature)

    allowed_keys = _allowed_argument_names(schema=schema, properties=properties, signature=signature)
    normalized: dict[str, Any] = {}
    for key, value in args.items():
        if allowed_keys is not None and key not in allowed_keys:
            continue
        coerced_value = _coerce_schema_value(value, properties.get(key))
        if coerced_value is None and key not in required_keys:
            continue
        normalized[key] = coerced_value

    missing = [key for key in sorted(required_keys) if key not in normalized]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"invalid arguments for tool '{tool_name}': missing required {joined}")

    errors: list[str] = []
    for key, value in normalized.items():
        errors.extend(_validate_schema_value(value, properties.get(key), key))
    if errors:
        raise ValueError(f"invalid arguments for tool '{tool_name}': {'; '.join(errors)}")

    return normalized


def _allowed_argument_names(
    *,
    schema: dict[str, Any],
    properties: dict[str, Any],
    signature: inspect.Signature | None,
) -> set[str] | None:
    signature_names, allows_var_kwargs = _signature_parameter_names(signature)
    schema_names = set(properties)

    if schema.get("additionalProperties") is True:
        if signature_names and not allows_var_kwargs:
            return signature_names
        return None

    if schema_names or signature_names:
        return schema_names | signature_names
    if allows_var_kwargs:
        return None
    return signature_names or None


def _signature_parameter_names(signature: inspect.Signature | None) -> tuple[set[str], bool]:
    if signature is None:
        return set(), False

    names: set[str] = set()
    allows_var_kwargs = False
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            allows_var_kwargs = True
            continue
        if parameter.kind in _REQUIRED_PARAM_KINDS:
            names.add(name)
    return names, allows_var_kwargs


def _required_argument_names(
    schema: dict[str, Any],
    signature: inspect.Signature | None,
) -> set[str]:
    required = {
        str(name)
        for name in schema.get("required", [])
        if isinstance(name, str) and name
    }
    if signature is None:
        return required

    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.kind not in _REQUIRED_PARAM_KINDS:
            continue
        if parameter.default is inspect.Signature.empty:
            required.add(name)
    return required


def _coerce_schema_value(value: Any, schema: dict[str, Any] | None) -> Any:
    if not isinstance(schema, dict):
        return value

    schema_type = schema.get("type")
    if schema_type == "string":
        if isinstance(value, str):
            return value
        if value is None or isinstance(value, (dict, list)):
            return value
        return str(value)

    if schema_type == "integer":
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                try:
                    return int(stripped)
                except ValueError:
                    return value
        return value

    if schema_type == "number":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                try:
                    return float(stripped)
                except ValueError:
                    return value
        return value

    if schema_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return value

    if schema_type == "array":
        items_schema = schema.get("items")
        if isinstance(value, list):
            return [_coerce_schema_value(item, items_schema) for item in value]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return value
            if isinstance(parsed, list):
                return [_coerce_schema_value(item, items_schema) for item in parsed]
        return value

    if schema_type == "object":
        properties = schema.get("properties")
        nested_properties = properties if isinstance(properties, dict) else {}
        if isinstance(value, dict):
            return {
                key: _coerce_schema_value(item, nested_properties.get(key))
                for key, item in value.items()
            }
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return value
            if isinstance(parsed, dict):
                return {
                    key: _coerce_schema_value(item, nested_properties.get(key))
                    for key, item in parsed.items()
                }
        return value

    return value


def _validate_schema_value(
    value: Any,
    schema: dict[str, Any] | None,
    label: str,
) -> list[str]:
    if not isinstance(schema, dict):
        return []

    schema_type = schema.get("type")
    errors: list[str] = []
    if schema_type == "string":
        if not isinstance(value, str):
            return [f"{label} should be string"]
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        if isinstance(min_length, int) and len(value) < min_length:
            errors.append(f"{label} must be at least {min_length} chars")
        if isinstance(max_length, int) and len(value) > max_length:
            errors.append(f"{label} must be at most {max_length} chars")
    elif schema_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return [f"{label} should be integer"]
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{label} must be >= {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"{label} must be <= {maximum}")
    elif schema_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [f"{label} should be number"]
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{label} must be >= {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"{label} must be <= {maximum}")
    elif schema_type == "boolean":
        if not isinstance(value, bool):
            return [f"{label} should be boolean"]
    elif schema_type == "array":
        if not isinstance(value, list):
            return [f"{label} should be array"]
        item_schema = schema.get("items")
        for index, item in enumerate(value):
            errors.extend(_validate_schema_value(item, item_schema, f"{label}[{index}]"))
    elif schema_type == "object":
        if not isinstance(value, dict):
            return [f"{label} should be object"]
        properties = schema.get("properties")
        nested_properties = properties if isinstance(properties, dict) else {}
        for key in schema.get("required", []):
            if isinstance(key, str) and key not in value:
                errors.append(f"missing required {label}.{key}")
        for key, item in value.items():
            errors.extend(
                _validate_schema_value(
                    item,
                    nested_properties.get(key),
                    f"{label}.{key}",
                )
            )

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors.append(f"{label} must be one of {enum_values}")
    return errors


class ToolRegistry:
    """Discover tools from the `hannah.tools` package."""

    def __init__(self, tools_dir: str | Path | None = None) -> None:
        self.tools_dir = Path(tools_dir or Path(__file__).resolve().parents[1] / "tools")
        self._tools = self._discover()

    def _discover(self) -> dict[str, RegisteredTool]:
        tools: dict[str, RegisteredTool] = {}
        for tool_path in sorted(self.tools_dir.glob("*/tool.py")):
            module_name = ".".join(tool_path.relative_to(self.tools_dir.parent.parent).with_suffix("").parts)
            try:
                module = importlib.import_module(module_name)
            except Exception as err:
                console.print(f"[yellow]warning:[/yellow] failed to import {module_name}: {err}")
                continue

            skill = getattr(module, "SKILL", None)
            run_fn = getattr(module, "run", None)
            if not isinstance(skill, dict) or run_fn is None:
                console.print(f"[yellow]warning:[/yellow] skipping malformed tool {module_name}")
                continue

            name = str(skill.get("name", tool_path.parent.name))
            parameters = skill.get("parameters", {"type": "object", "properties": {}})
            if not isinstance(parameters, dict):
                parameters = {"type": "object", "properties": {}}
            tools[name] = RegisteredTool(
                name=name,
                description=str(skill.get("description", "")),
                module_name=module_name,
                module=module,
                parameters=parameters,
                run_fn=run_fn,
                signature=inspect.signature(run_fn),
            )
        return tools

    def get_tool_specs(self) -> list[dict]:
        specs: list[dict] = []
        for tool in self._tools.values():
            parameters = deepcopy(tool.parameters)
            if parameters.get("type") == "object" and "additionalProperties" not in parameters:
                parameters["additionalProperties"] = False
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": parameters,
                    },
                }
            )
        return specs

    def normalize_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"unknown tool: {name}")
        normalized_input = args
        normalizer = getattr(tool.module, "normalize_args", None)
        if callable(normalizer) and isinstance(args, dict):
            normalized_input = normalizer(dict(args))
            if not isinstance(normalized_input, dict):
                raise TypeError(f"tool {name} normalize_args returned {type(normalized_input).__name__}, expected dict")
        return normalize_tool_args(
            name,
            normalized_input,
            parameters=tool.parameters,
            signature=tool.signature,
        )

    async def call(self, name: str, args: dict) -> dict:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"unknown tool: {name}")
        normalized_args = self.normalize_args(name, args)
        result = tool.run_fn(**normalized_args)
        if inspect.isawaitable(result):
            resolved = await result
        else:
            resolved = result
        if not isinstance(resolved, dict):
            raise TypeError(f"tool {name} returned {type(resolved).__name__}, expected dict")
        return resolved

    def list_tools(self) -> None:
        table = Table(title="Registered Hannah Tools")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Module", style="dim")
        for tool in self._tools.values():
            table.add_row(tool.name, tool.description, tool.module_name)
        console.print(table)
