"""Shared schema payload and local-diff helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, cast

from .types import (
    SchemaAttributeChange,
    SchemaDiff,
    SchemaEntityType,
    SchemaSearchSupport,
    SchemaTableDiff,
)


def strip_entity_text(schema: Any) -> Any:
    """Return a shallow structural copy without generated ``entityText``.

    Schema dictionaries may carry arbitrary application metadata. Copying only
    the root, entity list, and entity mappings preserves that open contract and
    avoids mutating caller-owned schema objects.
    """

    if not isinstance(schema, Mapping):
        return schema
    result = dict(schema)
    entities = result.get("entities")
    if isinstance(entities, list):
        copied_entities = []
        for entity in entities:
            if isinstance(entity, Mapping):
                copied_entity = dict(entity)
                copied_entity.pop("entityText", None)
                copied_entities.append(copied_entity)
            else:
                copied_entities.append(entity)
        result["entities"] = copied_entities
    return result


def _entities_by_name(schema: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(schema, Mapping):
        return {}
    entities = schema.get("entities")
    if not isinstance(entities, list):
        return {}
    return {
        entity["name"]: dict(entity)
        for entity in entities
        if isinstance(entity, Mapping) and isinstance(entity.get("name"), str) and entity["name"]
    }


def _effective_entity_type(entity: Mapping[str, Any]) -> SchemaEntityType:
    value = entity.get("type")
    return "DEFAULT" if value is None else cast(SchemaEntityType, value)


def _effective_search_support(entity: Mapping[str, Any]) -> SchemaSearchSupport:
    value = entity.get("searchSupport")
    return "BOTH" if value is None else cast(SchemaSearchSupport, value)


def _attributes_by_name(entity: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    attributes = entity.get("attributes")
    if not isinstance(attributes, list):
        return {}
    return {
        attribute["name"]: dict(attribute)
        for attribute in attributes
        if isinstance(attribute, Mapping)
        and isinstance(attribute.get("name"), str)
        and attribute["name"]
    }


def _other_entity_fields(entity: Mapping[str, Any]) -> Dict[str, Any]:
    ignored = {"name", "type", "searchSupport", "attributes", "entityText"}
    normalized = {key: value for key, value in entity.items() if key not in ignored}

    # The Cloud schema serializer may either omit optional collections or emit
    # empty arrays. Those representations describe the same entity.
    for key in ("indexes", "resolvers", "triggers"):
        if normalized.get(key) is None:
            normalized[key] = []

    partition = normalized.get("partition")
    if partition is None:
        normalized["partition"] = ""
    elif isinstance(partition, str):
        normalized["partition"] = partition.strip()

    # Missing and explicit null identifiers are equivalent. An empty object is
    # retained so invalid/incomplete local definitions still appear in a diff.
    if normalized.get("identifier") is None:
        normalized["identifier"] = None

    return normalized


def compute_schema_diff(remote_schema: Any, local_schema: Any) -> SchemaDiff:
    """Compare schemas with Cloud's backward-compatible search defaults."""

    remote_entities = _entities_by_name(strip_entity_text(remote_schema))
    local_entities = _entities_by_name(strip_entity_text(local_schema))
    added = sorted(name for name in local_entities if name not in remote_entities)
    removed = sorted(name for name in remote_entities if name not in local_entities)
    changed_tables: list[SchemaTableDiff] = []

    for name in sorted(local_entities):
        if name not in remote_entities:
            continue
        local_entity = local_entities[name]
        remote_entity = remote_entities[name]
        table_diff: SchemaTableDiff = {"name": name}

        type_from = _effective_entity_type(remote_entity)
        type_to = _effective_entity_type(local_entity)
        if type_from != type_to:
            table_diff["type"] = {"from": type_from, "to": type_to}

        if type_from == "SEARCHABLE" or type_to == "SEARCHABLE":
            support_from = _effective_search_support(remote_entity)
            support_to = _effective_search_support(local_entity)
            if support_from != support_to:
                table_diff["searchSupport"] = {
                    "from": support_from,
                    "to": support_to,
                }

        local_attributes = _attributes_by_name(local_entity)
        remote_attributes = _attributes_by_name(remote_entity)
        added_attributes = sorted(name for name in local_attributes if name not in remote_attributes)
        removed_attributes = sorted(name for name in remote_attributes if name not in local_attributes)
        changed_attributes: list[SchemaAttributeChange] = []
        for attribute_name in sorted(local_attributes):
            if attribute_name not in remote_attributes:
                continue
            local_attribute = local_attributes[attribute_name]
            remote_attribute = remote_attributes[attribute_name]
            if local_attribute != remote_attribute:
                changed_attributes.append(
                    {
                        "name": attribute_name,
                        "from": remote_attribute,
                        "to": local_attribute,
                    }
                )
        if added_attributes or removed_attributes or changed_attributes:
            table_diff["attributes"] = {
                "added": added_attributes,
                "removed": removed_attributes,
                "changed": changed_attributes,
            }

        # Existing async diff behavior detected changes to indexes, resolvers,
        # triggers, identifiers, and arbitrary extension metadata. Retain that
        # detection even though the compact Python diff does not expand those
        # fields into per-property details yet.
        other_fields_changed = _other_entity_fields(remote_entity) != _other_entity_fields(local_entity)
        if len(table_diff) > 1 or other_fields_changed:
            changed_tables.append(table_diff)

    return {
        "added_tables": added,
        "removed_tables": removed,
        "changed_tables": changed_tables,
    }
