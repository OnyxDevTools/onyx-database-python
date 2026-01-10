# Codegen optional Pydantic models

**Gap**: Python codegen emits plain classes only; no optional validation layer analogous to TS type safety.

**Plan**
- Add `--models pydantic` flag to `onyx-py gen` to emit Pydantic models (still allow current plain class mode as default).
- Preserve extra/unknown fields allowance for resolver data.
- Update docs and examples to show both modes; add small validation tests.

**Reference (TypeScript)**: Type-level safety via generated interfaces in `onyx-database-typescript/src/helpers/types.ts` (codegen in `onyx-database-typescript/gen/*`).
