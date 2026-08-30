"""Lightweight shared type aliases used across the SDK."""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Literal,
    NotRequired,
    Optional,
    Protocol,
    Required,
    Sequence,
    TypeAlias,
    TypedDict,
    Union,
)


class QueryBuilderLike(Protocol):
    def to_query_object(self) -> Dict[str, Any]:
        ...


Condition = Dict[str, Any]
Sort = Dict[str, str]

SearchMode: TypeAlias = Literal["lexical", "semantic", "hybrid"]
SearchMatch: TypeAlias = Literal["all", "any"]


class SearchOptions(TypedDict, total=False):
    """Options for the high-level natural-language ``search`` API.

    Python mappings may use snake_case or the Cloud wire camelCase aliases. Do
    not supply both spellings of the same option.
    """

    mode: SearchMode
    match: SearchMatch
    min_score: Optional[float]
    max_candidates: int
    minScore: Optional[float]
    maxCandidates: int


QueryCriteriaOperator: TypeAlias = Literal[
    "EQUAL",
    "NOT_EQUAL",
    "IN",
    "NOT_IN",
    "GREATER_THAN",
    "GREATER_THAN_EQUAL",
    "LESS_THAN",
    "LESS_THAN_EQUAL",
    "MATCHES",
    "NOT_MATCHES",
    "BETWEEN",
    "NOT_BETWEEN",
    "LIKE",
    "NOT_LIKE",
    "CONTAINS",
    "CONTAINS_IGNORE_CASE",
    "NOT_CONTAINS",
    "NOT_CONTAINS_IGNORE_CASE",
    "STARTS_WITH",
    "NOT_STARTS_WITH",
    "IS_NULL",
    "NOT_NULL",
    "CANDIDATES",
    "SEARCH_CANDIDATES",
    "HNSW_CANDIDATES",
    "SEARCH",
]


class FullTextQuery(TypedDict):
    """Legacy-compatible exact full-text search payload."""

    queryText: str
    minScore: Optional[float]


Int64WireInput: TypeAlias = Union[int, str]


class SemanticVectorSignature(TypedDict):
    """Canonical lossless semantic routing signature sent over the wire."""

    calibrationId: str
    bucketId: int
    cells: List[int]
    cellCounts: List[int]
    fingerprint: List[str]
    bands: List[str]
    boundaryConfidence: float


class SemanticVectorSignatureInput(TypedDict, total=False):
    """Input accepted by ``semantic_vector_signature``.

    Field names match the Cloud JSON contract. The helper also accepts snake_case
    aliases when a plain mapping is supplied.
    """

    calibrationId: Required[Int64WireInput]
    bucketId: Required[int]
    cells: Required[Sequence[int]]
    cellCounts: Required[Sequence[int]]
    fingerprint: Required[Sequence[Int64WireInput]]
    bands: NotRequired[Sequence[Int64WireInput]]
    boundaryConfidence: NotRequired[float]


class VectorSearchQuery(TypedDict):
    """Canonical native lexical, semantic, or hybrid search payload."""

    text: Optional[str]
    semantic: Optional[SemanticVectorSignature]
    minScore: Optional[float]
    nearbyBucketRadius: int
    maxCandidates: int
    requireAllTerms: bool


class VectorSearchQueryInput(TypedDict, total=False):
    """Input accepted by ``vector_search_query``."""

    text: Optional[str]
    semantic: Optional[SemanticVectorSignatureInput]
    minScore: Optional[float]
    nearbyBucketRadius: int
    maxCandidates: int
    requireAllTerms: bool


class HnswSearchQuery(TypedDict):
    """Canonical bounded native-HNSW candidate request."""

    calibrationId: str
    vector: List[float]
    maxCandidates: int
    efSearch: int
    minScore: Optional[float]
    formatVersion: Literal[1]


class HnswSearchQueryInput(TypedDict, total=False):
    """Input accepted by ``hnsw_search_query``."""

    calibrationId: Required[Int64WireInput]
    vector: Required[Sequence[float]]
    maxCandidates: int
    efSearch: int
    minScore: Optional[float]
    formatVersion: int


class ApproximateIndexCandidateQuery(TypedDict):
    """Bounded ordinary-index candidate route."""

    values: List[Any]
    maxCandidates: int


class ApproximateSearchOptions(TypedDict, total=False):
    """Options for text-only ``approximate_search`` calls."""

    minScore: Optional[float]
    maxCandidates: int
    requireAllTerms: bool


class QueryPage(TypedDict, total=False):
    items: List[Any]
    next_page: Optional[str]
    total_count: Optional[int]


class StreamHandlers(TypedDict, total=False):
    on_item_added: Optional[callable]
    on_item_updated: Optional[callable]
    on_item_deleted: Optional[callable]
    on_item: Optional[callable]


SchemaEntityType: TypeAlias = Literal["DEFAULT", "SEARCHABLE"]
SchemaSearchSupport: TypeAlias = Literal["LEXICAL", "SEMANTIC", "BOTH"]


class SchemaIdentifier(TypedDict, total=False):
    """Primary-key definition in an Onyx Cloud schema entity."""

    name: Required[str]
    generator: str
    type: str


class SchemaAttribute(TypedDict, total=False):
    """Attribute definition in an Onyx Cloud schema entity."""

    name: Required[str]
    type: Required[str]
    isNullable: bool


class SchemaIndex(TypedDict, total=False):
    """Index definition in an Onyx Cloud schema entity."""

    name: Required[str]
    type: str
    minimumScore: float


class SchemaResolver(TypedDict, total=False):
    """Resolver definition in an Onyx Cloud schema entity."""

    name: Required[str]
    resolver: Required[str]


class SchemaTrigger(TypedDict, total=False):
    """Trigger definition in an Onyx Cloud schema entity."""

    name: Required[str]
    event: Required[str]
    trigger: Required[str]


class SchemaEntity(TypedDict, total=False):
    """Typed schema entity while retaining raw-dict schema API compatibility.

    ``searchSupport`` applies to ``SEARCHABLE`` entities. Omitting it is
    equivalent to ``"BOTH"`` for backward compatibility.
    """

    name: Required[str]
    type: SchemaEntityType
    searchSupport: SchemaSearchSupport
    identifier: SchemaIdentifier
    partition: str
    attributes: List[SchemaAttribute]
    indexes: List[SchemaIndex]
    resolvers: List[SchemaResolver]
    triggers: List[SchemaTrigger]
    entityText: str


class SchemaUpsertRequest(TypedDict, total=False):
    """Schema payload accepted by validation and update operations."""

    databaseId: str
    revisionDescription: str
    entities: Required[List[SchemaEntity]]


SchemaInput: TypeAlias = Union[SchemaUpsertRequest, Dict[str, Any]]
SchemaEntityTypeChange = TypedDict(
    "SchemaEntityTypeChange",
    {"from": SchemaEntityType, "to": SchemaEntityType},
)
SchemaSearchSupportChange = TypedDict(
    "SchemaSearchSupportChange",
    {"from": SchemaSearchSupport, "to": SchemaSearchSupport},
)


SchemaAttributeChange = TypedDict(
    "SchemaAttributeChange",
    {"name": str, "from": Dict[str, Any], "to": Dict[str, Any]},
)


class SchemaAttributeDiff(TypedDict):
    added: List[str]
    removed: List[str]
    changed: List[SchemaAttributeChange]


class SchemaTableDiff(TypedDict, total=False):
    name: Required[str]
    type: SchemaEntityTypeChange
    searchSupport: SchemaSearchSupportChange
    attributes: SchemaAttributeDiff


class SchemaDiff(TypedDict, total=False):
    added_tables: List[str]
    removed_tables: List[str]
    changed_tables: List[SchemaTableDiff]
