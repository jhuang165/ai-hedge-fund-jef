# ai-hedge-fund-jef

Project guide for AI coding agents working in `/Users/programming-xcode/ai-hedge-fund-jef`.

## Shared memory bridge

This repository uses the Mac-local shared memory bridge (project id `ai-hedge-fund`); Claude
native auto-memory and Codex native memories are disabled. At every session start, run the
exact `memoryctl load` command emitted by the SessionStart hook with its current writer and
session ID, read the returned `MEMORY.md` index in full, then read only relevant bodies with
`memoryctl read --repo "$PWD" --topic TOPIC-SLUG`. Treat that revision-bound load as required
context; reload before continuing if a hook reports newer memory.

After every meaningful completed unit and before stopping or compacting, start one focused
lowercase kebab-case topic with `memoryctl begin --repo "$PWD" --topic TOPIC-SLUG --operation
upsert --writer WRITER --session SESSION-ID`. Edit only the staged `topic.md` and `pointer.txt`,
never the canonical memory directory. Record the change and reason, absolute date, state,
branch, key files or functions, verification, pending work, and non-obvious decisions; keep the
pointer to exactly one Markdown link and exclude secrets and transient narration. Commit with
`memoryctl commit TRANSACTION-ID`. On a stale revision, reload, reread, and reconcile through a
new transaction, never force-copy staged files. Use `memoryctl no-op` only for a clean session
with a truthful reason.

The operator guide is `~/claude-tools/docs/shared-memory-bridge.md`.
