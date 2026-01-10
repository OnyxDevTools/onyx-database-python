# CLI parity and polish

**Gap**: CLI lacks some TS niceties (subset-to-stdout for gen, retries/timeout flags, consistent output).

**Plan**
- Add flags for timeout/retry to `onyx-py schema`/`gen` (aligned with HTTP config).
- Add subset-to-stdout behavior for `gen` similar to TS `onyx-gen --tables` printing.
- Improve formatting/help consistency and error handling (non-zero exits on failures).
- Add tests or golden outputs for CLI commands where feasible.
- Document any changes or new features in the README.md ## CLI (codegen + schema) section 

**Reference (TypeScript)**: `onyx-database-typescript/gen/cli/generate.ts`, `onyx-database-typescript/gen/cli/schema.ts`
