## Code graph (cbm tools)

The cbm tools query a prebuilt code graph of the indexed repositories. They answer structural questions in one
small result, so a long session stays short. Use them before reading files:

- `project` is required: pass the repository's absolute path (or its folder name). `list_projects` shows the
  indexed ones.
- "Where is X defined" or "what is X": `search_graph` by name, then `get_code_snippet` for just that function.
  Don't read the whole file.
- "Who calls X" or "what does X call", and request flows across files: `trace_path` (callers or callees, a small
  depth first).
- A large unfamiliar file: `get_file_outline` first, then read only the lines you need.
- Orientation in a new repository: `get_architecture` once.
- Fall back to grep and read when a graph query returns nothing, for string literals, config values, or files the
  graph does not cover. Before you conclude that something does not exist, run `check_index_coverage` or a grep.
- The graph reflects the last index. If `index_status` says it is stale, or the code was just edited, confirm with
  a range read.
