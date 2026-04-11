# Token Efficiency and Engineering Rules

## Core policy
Be extremely token-efficient.
Prefer minimal context, minimal output, and narrow tool usage.
Do not read, print, or summarize large amounts of text unless strictly
necessary.

## Output rules
- Default to concise answers.
- Give the result first.
- Do not restate the task.
- Do not add background unless it changes the decision.
- Keep prose compact and information-dense.
- When possible, answer in bullets of at most 5 items.
- For code changes, show only the relevant diff or the exact edited block.
- Do not print unchanged code.
- Do not quote large file contents.

## Documentation-first rules
- When adding a new model, read the relevant documentation first.
- Do not implement a new model based on assumptions when authoritative docs
  are available.
- Before integrating a model, verify its API, required inputs, outputs,
  configuration, and performance-relevant constraints from the docs.
- If the documentation is ambiguous, state the uncertainty explicitly instead
  of guessing.

## API design rules
- When adding a new feature, prefer a shared common API rather than a
  model-specific ad hoc implementation.
- Do not introduce model-specific divergence unless it is strictly necessary.
- Shared APIs must not impose avoidable runtime overhead.
- Preserve execution performance. Abstractions must be designed so that hot
  paths remain efficient.
- Prefer common infrastructure for shared behavior, but do not degrade latency,
  throughput, memory use, or compile behavior merely for uniformity.
- If a model requires a specialized fast path, keep the external API common
  while allowing an optimized backend implementation.

## File reading rules
- Never read whole files unless necessary.
- First inspect file size and structure with narrow tools.
- Prefer targeted reads over full reads.
- Read only the smallest relevant slice of a file.
- If a file is large, first identify the exact function, class, or line range
  needed.
- Do not reread the same file content repeatedly.
- If content was already inspected, reuse that information.

## Search rules
- Prefer `rg` or targeted search over broad file reads.
- Search narrowly by symbol, function name, error string, or path.
- Avoid searching the entire repository when a subdirectory is enough.
- When multiple candidate files exist, inspect the most likely one first.

## Command rules
- Use commands that minimize output.
- Prefer quiet flags and narrow paths.
- Prefer:
  - `rg pattern path`
  - `git diff -- path/to/file`
  - `sed -n 'start,endp' file`
  - `head`
  - `tail`
  - targeted test invocation
- Avoid:
  - full recursive listings
  - full test suite runs unless required
  - verbose build logs unless debugging requires them

## Editing rules
- Make the smallest viable change.
- Edit only the files directly relevant to the task.
- Do not refactor unrelated code.
- Do not propose broad cleanup unless requested.

## Testing rules
- Run the narrowest test that can validate the change.
- Expand test scope only if the narrow test fails in a way that requires it.
- Do not run full suites by default.

## Git and diff rules
- Inspect diffs narrowly.
- Prefer file-scoped diff, word diff, or hunk-level inspection.
- Summarize changes in one short list.
- Do not paste large diffs unless asked.

## Logging rules
- If command output is long, summarize it instead of reproducing it.
- Extract only the error lines or the decisive lines.
- Do not include installation logs, lockfile noise, or repetitive stack traces
  unless necessary.

## Decision rules
Before using a tool or reading a file, ask:
1. What is the minimum information needed?
2. What is the narrowest command that gets it?
3. Can I answer with a shorter output format?

Before implementing a model or feature, ask:
1. What do the docs say?
2. Can this be expressed through the common API?
3. How do I avoid adding runtime overhead?

## Interaction rules
- Ask at most one clarifying question, and only if it prevents wasted work.
- If assumptions are safe, proceed with the most likely one.
- Prefer action over discussion.
- Keep updates short and only when they add new information.

## For repository work
When investigating code:
1. identify the likely file or symbol
2. search narrowly
3. read only the relevant lines
4. patch minimally
5. run the narrowest validation
6. report only the essential result

When adding a model:
1. read the official documentation first
2. verify the integration points and constraints
3. implement through the common API where appropriate
4. preserve the optimized execution path
5. validate behavior with narrow tests

## Hard constraints
- Do not dump large file contents.
- Do not produce long tutorials unless requested.
- Do not read whole generated files, lockfiles, or vendored code.
- Do not use broad scans when a targeted query is enough.
- Do not be verbose.
- Do not add a model without consulting the docs.
- Do not add feature behavior through model-specific ad hoc code when a shared
  API is appropriate.
- Do not sacrifice execution performance for conceptual cleanliness alone.

## Engineering baseline
- All source code, comments, and docstrings must be in English.
- Follow the repository coding standards in `AGENTS.md`.
- Keep implementations clear, explicit, typed, and testable.