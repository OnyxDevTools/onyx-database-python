"""Onyx Database client facade."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

from .ai import iter_sse
from .config import ResolvedConfig, clear_config_cache, resolve_config
from .errors import OnyxHTTPError
from .http import HttpClient, serialize_dates
from .query_builder import QueryBuilder
from .stream import open_json_lines_stream
from .types import SchemaDiff


class _Cascade:
    def __init__(self, db: "OnyxDatabase", relationships: Iterable[str]):
        self._db = db
        self._relationships = [r for r in relationships if r]

    def save(self, table: str, entity_or_entities: Any) -> Any:
        return self._db.save(table, entity_or_entities, {"relationships": self._relationships})

    def delete(self, table: str, primary_key: str, **options: Any) -> Any:
        opts = dict(options)
        opts["relationships"] = self._relationships
        return self._db.delete(table, primary_key, opts)


class _AiChatClient:
    def __init__(self, db: "OnyxDatabase"):
        self._db = db

    def create(
        self,
        request: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None,
        *,
        database_id: Optional[str] = None,
    ) -> Any:
        if options is None:
            opts: Dict[str, Any] = {}
        elif isinstance(options, dict):
            opts = dict(options)
        else:
            raise TypeError("options must be a dict when provided")
        if database_id is None:
            database_id = opts.get("database_id") or opts.get("databaseId")
        return self._db._chat_request(request, database_id=database_id)


class _AiNamespace:
    def __init__(self, db: "OnyxDatabase"):
        self._db = db
        self._chat_client = _AiChatClient(db)

    def chat_client(self) -> _AiChatClient:
        return self._chat_client

    def chat(
        self,
        content_or_request: Any,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(content_or_request, str):
            return self._db._chat_with_content(content_or_request, options, **kwargs)
        if isinstance(content_or_request, dict):
            return self._chat_client.create(content_or_request, options, **kwargs)
        raise TypeError("chat expects a string prompt or request dict")

    def get_models(self) -> Any:
        return self._db.get_models()

    def get_model(self, model_id: str) -> Any:
        return self._db.get_model(model_id)

    def request_script_approval(self, script: Any) -> Any:
        if isinstance(script, dict):
            script_value = script.get("script")
        else:
            script_value = script
        if not script_value:
            raise ValueError("script is required")
        return self._db.request_script_approval(str(script_value))


class OnyxDatabase:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        model_map = config.pop("model_map", None)
        schema_models = config.pop("schema", None)
        models = config.pop("models", None)

        self._model_map = model_map or {}

        # Prefer explicit schema mapping (dict or module with SCHEMA / MODEL_MAP)
        if not self._model_map and schema_models:
            if isinstance(schema_models, dict):
                self._model_map = schema_models
            else:
                candidate = getattr(schema_models, "SCHEMA", None) or getattr(schema_models, "MODEL_MAP", None)
                if isinstance(candidate, dict):
                    self._model_map = candidate

        # Fallback to module with MODEL_MAP attr
        if not self._model_map and models:
            candidate = getattr(models, "MODEL_MAP", None)
            if isinstance(candidate, dict):
                self._model_map = candidate

        self._config_input = config
        self._resolved: ResolvedConfig = resolve_config(self._config_input)
        self._http = HttpClient(
            self._resolved.base_url,
            self._resolved.api_key,
            self._resolved.api_secret,
            request_logging_enabled=self._resolved.request_logging_enabled,
            response_logging_enabled=self._resolved.response_logging_enabled,
            request_timeout_seconds=self._resolved.request_timeout_seconds,
            max_retries=self._resolved.max_retries,
            retry_backoff_seconds=self._resolved.retry_backoff_seconds,
        )
        self._ai_http = HttpClient(
            self._resolved.ai_base_url,
            self._resolved.api_key,
            self._resolved.api_secret,
            request_logging_enabled=self._resolved.request_logging_enabled,
            response_logging_enabled=self._resolved.response_logging_enabled,
            request_timeout_seconds=self._resolved.request_timeout_seconds,
            max_retries=self._resolved.max_retries,
            retry_backoff_seconds=self._resolved.retry_backoff_seconds,
        )
        self._base_url = self._resolved.base_url
        self._ai_base_url = self._resolved.ai_base_url
        self._database_id = self._resolved.database_id
        self._default_partition = self._resolved.partition
        self._default_model = self._resolved.default_model
        self.ai = _AiNamespace(self)

    def _strip_entity_text(self, schema: Any) -> Any:
        """Remove noisy entityText blocks from schemas (mutates in place)."""
        if not isinstance(schema, dict):
            return schema
        entities = schema.get("entities")
        if isinstance(entities, list):
            for ent in entities:
                if isinstance(ent, dict):
                    ent.pop("entityText", None)
        return schema

    # Entry points / builders
    def from_table(self, table: str) -> QueryBuilder:
        return QueryBuilder(self, table, partition=self._default_partition)

    def select(self, *fields) -> QueryBuilder:
        return QueryBuilder(self, None, partition=self._default_partition).select(*fields)

    def search(self, query_text: str, min_score: Optional[float] = None) -> QueryBuilder:
        return QueryBuilder(self, "ALL", partition=self._default_partition).search(query_text, min_score)

    def cascade(self, relationships: str) -> _Cascade:
        rels = [r.strip() for r in relationships.split(",")] if isinstance(relationships, str) else list(relationships)
        return _Cascade(self, rels)

    # CRUD helpers
    def save(self, table: str, entity_or_entities: Any, options: Optional[Dict[str, Any]] = None) -> Any:
        params = []
        opts = options or {}
        rels = opts.get("relationships") or []
        if rels:
            params.append(f"relationships={','.join(map(str, rels))}")
        query = f"?{'&'.join(params)}" if params else ""
        path = f"/data/{self._database_id}/{table}{query}"
        return self._http.request("PUT", path, serialize_dates(entity_or_entities))

    def batch_save(self, table: str, entities: Iterable[Any], batch_size: int = 1000, options: Optional[Dict[str, Any]] = None) -> None:
        chunk: list = []
        for entity in entities:
            chunk.append(entity)
            if len(chunk) >= batch_size:
                self.save(table, list(chunk), options)
                chunk = []
        if chunk:
            self.save(table, list(chunk), options)

    def find_by_id(self, table: str, primary_key: str, *, partition: Optional[str] = None, resolvers: Optional[Iterable[str]] = None) -> Any:
        params = []
        p = partition or self._default_partition
        if p:
            params.append(f"partition={p}")
        if resolvers:
            params.append(f"resolvers={','.join(resolvers)}")
        query = f"?{'&'.join(params)}" if params else ""
        path = f"/data/{self._database_id}/{table}/{primary_key}{query}"
        try:
            res = self._http.request("GET", path)
            return self._maybe_apply_model(table, res)
        except OnyxHTTPError as err:
            if err.status == 404:
                return None
            raise

    def delete(self, table: str, primary_key: str, options: Optional[Dict[str, Any]] = None) -> bool:
        params = []
        opts = options or {}
        p = opts.get("partition") or self._default_partition
        if p:
            params.append(f"partition={p}")
        rels = opts.get("relationships") or []
        if rels:
            params.append(f"relationships={','.join(rels)}")
        query = f"?{'&'.join(params)}" if params else ""
        path = f"/data/{self._database_id}/{table}/{primary_key}{query}"
        self._http.request("DELETE", path)
        return True

    # Query executor (used by QueryBuilder)
    def count(self, table: str, select: Dict[str, Any], partition: Optional[str]) -> int:
        params = []
        p = partition or self._default_partition
        if p:
            params.append(f"partition={p}")
        query = f"?{'&'.join(params)}" if params else ""
        path = f"/data/{self._database_id}/query/count/{table}{query}"
        return int(self._http.request("PUT", path, serialize_dates(select)))

    def query_page(self, table: str, select: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        params = []
        if options.get("pageSize") is not None:
            params.append(f"pageSize={options['pageSize']}")
        if options.get("nextPage"):
            params.append(f"nextPage={options['nextPage']}")
        partition = options.get("partition") or select.get("partition") or self._default_partition
        if partition:
            params.append(f"partition={partition}")
        query = f"?{'&'.join(params)}" if params else ""
        path = f"/data/{self._database_id}/query/{table}{query}"
        res = self._http.request("PUT", path, serialize_dates(select))
        if isinstance(res, dict) and "records" in res:
            return res
        return {"records": res or [], "nextPage": None}

    def delete_by_query(self, table: str, select: Dict[str, Any], partition: Optional[str]) -> Any:
        params = []
        p = partition or self._default_partition
        if p:
            params.append(f"partition={p}")
        query = f"?{'&'.join(params)}" if params else ""
        path = f"/data/{self._database_id}/query/delete/{table}{query}"
        return self._http.request("PUT", path, serialize_dates(select))

    def update(self, table: str, update_query: Dict[str, Any], partition: Optional[str]) -> Any:
        params = []
        p = partition or update_query.get("partition") or self._default_partition
        if p:
            params.append(f"partition={p}")
        query = f"?{'&'.join(params)}" if params else ""
        path = f"/data/{self._database_id}/query/update/{table}{query}"
        return self._http.request("PUT", path, serialize_dates(update_query))

    def stream(self, table: str, select: Dict[str, Any], include_query_results: bool, keep_alive: bool, handlers: Dict[str, Any]):
        params = []
        if include_query_results:
            params.append("includeQueryResults=true")
        if keep_alive:
            params.append("keepAlive=true")
        query = f"?{'&'.join(params)}" if params else ""
        hdrs = self._http.headers({"Accept": "application/x-ndjson", "Content-Type": "application/json"})
        body = json.dumps(serialize_dates(select))

        def opener():
            return self._http.open_stream(
                path=f"/data/{self._database_id}/query/stream/{table}{query}",
                method="PUT",
                body=body,
                headers=hdrs,
            )

        return open_json_lines_stream(opener, handlers=handlers)

    # Documents
    def save_document(self, doc: Dict[str, Any]) -> Any:
        path = f"/data/{self._database_id}/document"
        return self._http.request("PUT", path, serialize_dates(doc))

    def get_document(self, document_id: str, *, width: Optional[int] = None, height: Optional[int] = None) -> Any:
        params = []
        if width is not None:
            params.append(f"width={width}")
        if height is not None:
            params.append(f"height={height}")
        query = f"?{'&'.join(params)}" if params else ""
        path = f"/data/{self._database_id}/document/{document_id}{query}"
        return self._http.request("GET", path)

    def delete_document(self, document_id: str) -> Any:
        path = f"/data/{self._database_id}/document/{document_id}"
        return self._http.request("DELETE", path)

    # Schema APIs
    def get_schema(self, tables: Optional[Iterable[str]] = None) -> Any:
        params = []
        if tables:
            params.append(f"tables={','.join(tables)}")
        query = f"?{'&'.join(params)}" if params else ""
        path = f"/schemas/{self._database_id}{query}"
        res = self._http.request("GET", path)
        return self._strip_entity_text(res)

    def get_schema_history(self) -> Any:
        path = f"/schemas/history/{self._database_id}"
        return self._http.request("GET", path)

    def validate_schema(self, schema: Dict[str, Any]) -> Any:
        path = f"/schemas/{self._database_id}/validate"
        payload = self._strip_entity_text(dict(schema))
        return self._http.request("POST", path, serialize_dates(payload))

    def update_schema(self, schema: Dict[str, Any], *, publish: bool = False) -> Any:
        params = []
        if publish:
            params.append("publish=true")
        query = f"?{'&'.join(params)}" if params else ""
        payload = self._strip_entity_text(dict(schema))
        payload.setdefault("databaseId", self._database_id)
        path = f"/schemas/{self._database_id}{query}"
        return self._http.request("PUT", path, serialize_dates(payload))

    def diff_schema(self, local_schema: Dict[str, Any]) -> SchemaDiff:
        remote = self.get_schema()
        local_clean = self._strip_entity_text(dict(local_schema)) if isinstance(local_schema, dict) else {}
        remote_entities = {e.get("name"): e for e in remote.get("entities", [])} if isinstance(remote, dict) else {}
        local_entities = {e.get("name"): e for e in local_clean.get("entities", [])} if isinstance(local_clean, dict) else {}
        added = [name for name in local_entities.keys() if name not in remote_entities]
        removed = [name for name in remote_entities.keys() if name not in local_entities]

        def _attributes_by_name(entity: Dict[str, Any]):
            attrs = entity.get("attributes", []) if isinstance(entity, dict) else []
            return {a.get("name"): a for a in attrs if isinstance(a, dict) and a.get("name")}

        changed_tables = []
        for name in local_entities.keys():
            if name not in remote_entities:
                continue
            local_ent = local_entities[name]
            remote_ent = remote_entities[name]
            local_attrs = _attributes_by_name(local_ent)
            remote_attrs = _attributes_by_name(remote_ent)

            added_attrs = [n for n in local_attrs if n not in remote_attrs]
            removed_attrs = [n for n in remote_attrs if n not in local_attrs]
            changed_attrs = []
            for attr_name in local_attrs:
                if attr_name in remote_attrs:
                    l_attr = local_attrs[attr_name]
                    r_attr = remote_attrs[attr_name]
                    if json.dumps(l_attr, sort_keys=True) != json.dumps(r_attr, sort_keys=True):
                        changed_attrs.append({"name": attr_name, "from": r_attr, "to": l_attr})
            if added_attrs or removed_attrs or changed_attrs:
                changed_tables.append(
                    {
                        "name": name,
                        "attributes": {
                            "added": added_attrs,
                            "removed": removed_attrs,
                            "changed": changed_attrs,
                        },
                    }
                )

        return {"added_tables": added, "removed_tables": removed, "changed_tables": changed_tables}

    # Secrets
    def list_secrets(self) -> Any:
        path = f"/database/{self._database_id}/secret"
        return self._http.request("GET", path)

    def get_secret(self, key: str) -> Any:
        path = f"/database/{self._database_id}/secret/{key}"
        return self._http.request("GET", path, None, {"Content-Type": "application/json"})

    def put_secret(self, key: str, value: Dict[str, Any]) -> Any:
        path = f"/database/{self._database_id}/secret/{key}"
        return self._http.request("PUT", path, serialize_dates(value))

    def delete_secret(self, key: str) -> Any:
        path = f"/database/{self._database_id}/secret/{key}"
        return self._http.request("DELETE", path)

    def _merge_ai_options(
        self,
        options: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if options is None:
            merged: Dict[str, Any] = {}
        elif isinstance(options, dict):
            merged = dict(options)
        else:
            raise TypeError("options must be a dict when provided")
        if extra:
            for key, value in extra.items():
                if value is not None:
                    merged[key] = value
        for key, value in kwargs.items():
            if value is not None:
                merged[key] = value
        return merged

    def _extract_first_message_content(self, response: Any) -> str:
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message") or {}
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
        raise ValueError("Chat completion response is missing message content")

    def _chat_request(self, request: Dict[str, Any], *, database_id: Optional[str] = None) -> Any:
        payload = dict(request or {})
        if database_id is None:
            database_id = payload.pop("database_id", None) or payload.pop("databaseId", None)
        else:
            payload.pop("database_id", None)
            payload.pop("databaseId", None)
        messages = payload.get("messages")
        if messages is not None and not isinstance(messages, list):
            payload["messages"] = list(messages)
        stream = bool(payload.get("stream"))
        payload["stream"] = stream

        dbid = database_id if database_id is not None else self._database_id
        query = f"?databaseId={dbid}" if dbid else ""
        path = f"/v1/chat/completions{query}"
        if stream:
            body = json.dumps(serialize_dates(payload))
            headers = self._ai_http.headers({"Accept": "text/event-stream", "Content-Type": "application/json"})
            stream_resp = self._ai_http.open_stream(path=path, method="POST", body=body, headers=headers)
            return iter_sse(stream_resp)
        return self._ai_http.request("POST", path, serialize_dates(payload))

    def _chat_with_content(
        self,
        content: str,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        opts = self._merge_ai_options(options, **kwargs)
        raw = bool(opts.pop("raw", False))
        stream = bool(opts.pop("stream", False))
        role = opts.pop("role", None) or "user"
        model = opts.pop("model", None) or self._default_model
        database_id = opts.pop("database_id", None) or opts.pop("databaseId", None)
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": role, "content": content}],
            "stream": stream,
        }
        if "temperature" in opts:
            payload["temperature"] = opts.pop("temperature")
        if "top_p" in opts:
            payload["top_p"] = opts.pop("top_p")
        if "max_tokens" in opts:
            payload["max_tokens"] = opts.pop("max_tokens")
        if "metadata" in opts:
            payload["metadata"] = opts.pop("metadata")
        if "tools" in opts:
            payload["tools"] = opts.pop("tools")
        if "tool_choice" in opts:
            payload["tool_choice"] = opts.pop("tool_choice")
        if "user" in opts:
            payload["user"] = opts.pop("user")
        if opts:
            payload.update({k: v for k, v in opts.items() if v is not None})
        result = self._chat_request(payload, database_id=database_id)
        if stream:
            return result
        if raw:
            return result
        return self._extract_first_message_content(result)

    # AI endpoints
    def chat(
        self,
        content: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        *,
        messages: Optional[Iterable[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        database_id: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tools: Optional[Any] = None,
        tool_choice: Optional[Any] = None,
        user: Optional[str] = None,
        role: Optional[str] = None,
        raw: Optional[bool] = None,
        **extra: Any,
    ) -> Any:
        """Access AI chat: shorthand content or full request via chat client."""
        if (
            content is None
            and options is None
            and messages is None
            and model is None
            and stream is None
            and database_id is None
            and temperature is None
            and top_p is None
            and max_tokens is None
            and metadata is None
            and tools is None
            and tool_choice is None
            and user is None
            and role is None
            and raw is None
            and not extra
        ):
            return self.ai.chat_client()

        if isinstance(content, dict):
            if options is not None and not isinstance(options, dict):
                raise TypeError("options must be a dict when provided")
            return self.ai.chat(content, options, database_id=database_id)

        if isinstance(content, str):
            opts = self._merge_ai_options(
                options,
                extra=extra,
                database_id=database_id,
                model=model,
                role=role,
                stream=stream,
                raw=raw,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                metadata=metadata,
                tools=tools,
                tool_choice=tool_choice,
                user=user,
            )
            return self._chat_with_content(content, opts)

        if messages is None and model is None:
            raise TypeError("chat requires a prompt string, messages, or no arguments for the chat client")

        if messages is None or model is None:
            raise ValueError("chat requires both messages and model when using message-based calls")

        payload: Dict[str, Any] = {"model": model, "messages": list(messages)}
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if metadata is not None:
            payload["metadata"] = metadata
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if user is not None:
            payload["user"] = user
        payload.update({k: v for k, v in extra.items() if v is not None})
        payload["stream"] = bool(stream)
        return self._chat_request(payload, database_id=database_id)

    def get_models(self) -> Any:
        """List available Onyx AI models."""
        return self._ai_http.request("GET", "/v1/models")

    def get_model(self, model_id: str) -> Any:
        """Fetch metadata for a single Onyx AI model."""
        return self._ai_http.request("GET", f"/v1/models/{model_id}")

    def request_script_approval(self, script: str) -> Any:
        """Submit a script for mutation approval analysis."""
        return self._ai_http.request("POST", "/api/script-approvals", {"script": script})

    def clear(self) -> None:
        """Close streams; placeholder for future pooled resources."""
        pass

    def get_model_for_table(self, table: str):
        if isinstance(self._model_map, dict):
            return self._model_map.get(table)
        return None

    def _maybe_apply_model(self, table: str, value: Any) -> Any:
        model = self.get_model_for_table(table)
        if model is None or value is None:
            return value
        if isinstance(value, list):
            return [self._maybe_apply_model(table, v) for v in value]
        if isinstance(value, dict):
            return model(**value)
        return value


class OnyxFacade:
    def init(self, **config: Any) -> OnyxDatabase:
        return OnyxDatabase(config)

    def clear_cache_config(self) -> None:
        clear_config_cache()


onyx = OnyxFacade()
