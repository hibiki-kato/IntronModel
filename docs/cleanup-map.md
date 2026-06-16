# Cleanup Map

## Current Hotspots

- `src/tools/hparam_search.py`: largest Python file. Split by concern when
  touched next: config parsing, trial generation, process execution, result
  retention, and reporting.
- `src/tools/run_wrapper_pipeline.py`: backend for wrapper orchestration. Keep
  new wrapper behavior here or in `run/lib/`, not inside one-off shell code.
- `src/run_model.py`: public CLI. New model-specific CLI flags should enter via
  the model capability contract where possible.
- `run/*.sh`: user-facing config blocks. Runtime implementation should stay
  small and delegate to `run/lib/` or Python tools.

## Safe Cleanup Order

1. Remove generated artifacts only when Git shows them as ignored or untracked.
2. Move repeated shell helper logic into `run/lib/common.sh`.
3. Move repeated Python logic into `src/util/` with focused tests.
4. Update docs in the same change that moves or retires an entry point.
5. Archive legacy entry points before deleting them unless tests prove no caller
   still references them.

## Do Not Mix

- Do not combine model behavior changes with file moving.
- Do not combine data deletion with code refactors.
- Do not touch tracked user-edited wrapper configs unless the task requires it.
