"""Minimal schema-driven model generator (stdlib only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_name(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum() or ch == "_") or "Model"


def _timestamp_annotation(mode: str) -> str:
    if mode == "string":
        return "str"
    if mode == "number":
        return "float"
    return "datetime.datetime"


def _attribute_annotation(attr: Dict[str, Any], ts_mode: str) -> str:
    type_name = attr.get("type", "String")
    nullable = bool(attr.get("isNullable"))
    py = "str"
    if type_name.lower() in {"int", "integer"}:
        py = "int"
    elif type_name.lower() == "boolean":
        py = "bool"
    elif type_name.lower() in {"float", "double", "number"}:
        py = "float"
    elif type_name.lower() == "timestamp":
        py = _timestamp_annotation(ts_mode)
    elif type_name.lower() == "embeddedobject":
        py = "dict"
    ann = py
    if nullable:
        ann = f"Optional[{py}]"
    return ann


def _emit_tables(entities: List[Dict[str, Any]]) -> str:
    lines = ["class tables:", "    \"\"\"Table name constants.\"\"\""]
    if not entities:
        lines.append("    pass")
        return "\n".join(lines) + "\n"
    for e in entities:
        name = e.get("name", "")
        safe = _safe_name(name)
        lines.append(f"    {safe} = \"{name}\"")
    return "\n".join(lines) + "\n"


def _emit_init(entities: List[Dict[str, Any]]) -> str:
    class_names = [_safe_name(e.get("name", "Model")) for e in entities]
    imports = ", ".join(class_names)
    model_map_items = ", ".join([f'"{e.get("name", "")}": {cls}' for e, cls in zip(entities, class_names)])
    return "\n".join(
        [
            f"from .models import {imports}" if imports else "",
            "from .tables import tables",
            "from .schema import SCHEMA_JSON",
            f"SCHEMA = {{{model_map_items}}}",
            "__all__ = ['tables', 'SCHEMA_JSON', 'SCHEMA'" + (", " + ", ".join([f"'{c}'" for c in class_names]) if class_names else "") + "]",
        ]
    )


def _emit_model(entity: Dict[str, Any], ts_mode: str) -> str:
    name = _safe_name(entity.get("name", "Model"))
    attrs: List[Dict[str, Any]] = entity.get("attributes", []) or []
    lines = [
        f"class {name}:",
        '    """Generated model (plain Python class). Resolver/extra fields are allowed via **extra."""',
    ]
    init_params = ["self"]
    body: List[str] = []
    for attr in attrs:
        field = attr.get("name", "")
        ann = _attribute_annotation(attr, ts_mode)
        init_params.append(f"{field}: {ann} = None")
        body.append(f"        self.{field} = {field}")
    init_params.append("**extra: Any")
    body.append("        # allow resolver-attached fields or extra properties")
    body.append("        for k, v in extra.items():")
    body.append("            setattr(self, k, v)")
    if not body:
        body.append("        pass")
    lines.append(f"    def __init__({', '.join(init_params)}):")
    lines.extend(body)
    lines.append("")
    return "\n".join(lines)


def _emit_all_models(entities: List[Dict[str, Any]], ts_mode: str) -> str:
    out: List[str] = ["import datetime", "from typing import Any, Optional", ""]
    for e in entities:
        out.append(_emit_model(e, ts_mode))
        out.append("")
    return "\n".join(out)


def _emit_schema_map(schema: Dict[str, Any]) -> str:
    cleaned = {"databaseId": schema.get("databaseId"), "revisionDescription": schema.get("revisionDescription"), "entities": schema.get("entities", [])}
    py_literal = json.dumps(cleaned, indent=2)
    py_literal = (
        py_literal.replace("true", "True")
        .replace("false", "False")
        .replace("null", "None")
    )
    return "SCHEMA_JSON = " + py_literal + "\n"


def _emit_pydantic_models(entities: List[Dict[str, Any]], ts_mode: str) -> str:
    lines: List[str] = [
        "from pydantic import BaseModel, ConfigDict",
        "import datetime",
        "from typing import Optional, Any",
        "",
    ]
    for entity in entities:
        name = _safe_name(entity.get("name", "Model"))
        attrs: List[Dict[str, Any]] = entity.get("attributes", []) or []
        lines.append(f"class {name}(BaseModel):")
        if not attrs:
            lines.append("    model_config = ConfigDict(extra='allow')")
            lines.append("    pass")
            lines.append("")
            continue
        for attr in attrs:
            field = attr.get("name", "")
            ann = _attribute_annotation(attr, ts_mode)
            lines.append(f"    {field}: {ann} = None")
        lines.append("    model_config = ConfigDict(extra='allow')")
        lines.append("")
    return "\n".join(lines)


def generate_models(schema: Dict[str, Any], out: Path, *, package: Optional[str] = None, timestamp_mode: str = "datetime", models_mode: str = "plain") -> None:
    entities: List[Dict[str, Any]] = schema.get("entities", []) or []
    ts_mode = timestamp_mode
    if out.suffix == ".py":
        out.parent.mkdir(parents=True, exist_ok=True)
        model_block = _emit_all_models(entities, ts_mode) if models_mode == "plain" else _emit_pydantic_models(entities, ts_mode)
        content = "\n".join(
            [
                "import datetime",
                "from typing import Optional",
                "",
                _emit_tables(entities),
                _emit_schema_map(schema),
                model_block,
            ]
        )
        out.write_text(content, encoding="utf-8")
        return

    # treat as directory
    out.mkdir(parents=True, exist_ok=True)
    init_text = _emit_init(entities)
    (out / "__init__.py").write_text(init_text + "\n", encoding="utf-8")
    (out / "tables.py").write_text(_emit_tables(entities), encoding="utf-8")
    (out / "schema.py").write_text(_emit_schema_map(schema), encoding="utf-8")
    models_path = out / "models.py"
    model_block = _emit_all_models(entities, ts_mode) if models_mode == "plain" else _emit_pydantic_models(entities, ts_mode)
    models_path.write_text(model_block, encoding="utf-8")
