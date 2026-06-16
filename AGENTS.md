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

<claude-mem-context>
# Memory Context

# [IntronModel] recent context, 2026-05-19 9:47am EDT

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 23 obs (8,452t read) | 238,253t work | 96% savings

### May 19, 2026
116 1:18a 🔵 Gmail skill plugin cache missing at expected path
117 " 🔵 DNA AI Annotation Overleaf project structure mapped
118 " 🔵 DNA AI Annotation project purpose and file roles confirmed
119 1:19a 🔵 DNA annotation project email threads: competing methods, splice site entropy, ISMB submission, Hsap training
120 " 🔵 UniAnn preliminary results beat ANNEVO and Helixer on Drosophila chrX; AUGUSTUS still toughest competition
121 " 🔵 Splice site mononucleotide entropy analysis: annotated sites have HIGHER entropy than random — explained by exon GC balance
122 " 🔵 AlphaGenome splice-site score evaluation on Arabidopsis, Drosophila, Chicken — performance worse than Human/Chicken
123 1:20a ✅ Added "Recent project notes" section to main.tex with May 2026 email findings
124 " 🔵 latexmk compilation result unknown — zsh "read-only variable: status" prevented exit code capture
125 " ✅ main.tex "Recent project notes" section verified in place and PDF up-to-date
126 1:30a 🔵 CodonHunt DNA Annotation Project — RuleBook.tex structure mapped
127 1:31a 🔵 CodonHunt project files — current state of main.tex changelog and RuleBook gaps
128 " 🔵 DNA AI Annotation project — AGENTS.md file structure and bib coverage confirmed
129 1:32a 🟣 RuleBook.tex — "Known Methods" stubs replaced with full "Current annotation landscape" section
130 1:33a ✅ bib/hibiki.bib — four new annotation tool references added
131 " 🔴 bib/hibiki.bib new entries failed to apply via apply_patch — context mismatch
132 " 🔴 bib/hibiki.bib — four new entries successfully appended on retry with correct anchor
133 1:34a 🔵 RuleBook.tex latexmk build — compiles to 15-page PDF with warnings only, no errors
134 " 🔴 RuleBook.tex latexmk build fails — multiply defined label fig:placeholder
135 " 🔴 RuleBook.tex — duplicate fig:placeholder labels renamed to unique IDs
136 1:35a ✅ RuleBook.tex clean rebuild — 16-page PDF compiled successfully, no errors
137 1:36a 🔵 latexmk output path confusion — lualatex writes to root dir instead of .output/
138 " 🔵 RuleBook.tex content verified correct — all new bib entries confirmed in hibiki.bib, no LaTeX errors

Access 238k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>