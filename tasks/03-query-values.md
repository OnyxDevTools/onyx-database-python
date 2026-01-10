# Query helper parity: values()

**Gap**: Missing fluent `.values(field)` helper for query projections used in TS inner-query patterns.

**Plan**
- Add `.values(field)` to `QueryResults` (and async variant) returning list of field values.
- Ensure inner queries using `.values()` serialize correctly and examples cover it.
- Add tests for dict vs. model records.

**Reference (TypeScript)**: `onyx-database-typescript/src/builders/query-results.ts`
