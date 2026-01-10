# Async client and streams

**Gap**: No async client/streaming iterator ergonomics; TS supports async/await in Node/edge.

**Plan**
- Introduce `OnyxDatabaseAsync` with async HTTP client and async equivalents for save/query/stream.
- Provide async `QueryResults` with `async for` iteration and pagination.
- Keep API parity with sync client; factor shared logic to avoid duplication.
- Add async examples (queries, streams) and minimal tests using mocked async HTTP.

**Reference (TypeScript)**: `onyx-database-typescript/src/index.ts`, `onyx-database-typescript/src/builders/query-results.ts`, `onyx-database-typescript/src/http.ts`
