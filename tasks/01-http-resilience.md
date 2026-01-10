# HTTP resilience, timeouts, and error taxonomy

**Gap**: No configurable request timeout/backoff knobs; limited error taxonomy compared to TS SDK.

**Plan**
- Add timeout and retry/backoff options to config/env and thread into `HttpClient`.
- Broaden error classes (e.g., Unauthorized, NotFound, RateLimited, ServerError) and map HTTP status codes.
- Mirror request/response logging behaviors; keep defaults non-breaking.
- Add small unit tests (mocked HTTP) for retries, timeouts, and error mapping.

**Reference (TypeScript)**: `onyx-database-typescript/src/http.ts`, `onyx-database-typescript/src/errors.ts`
