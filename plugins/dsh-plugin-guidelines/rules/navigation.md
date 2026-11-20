## Code navigation

Most of your context and time goes into reading code. Navigate like this:

- Locate before you read: grep for the identifier, route or string (or glob for a file name) first, then read only the region around the hit with `offset` and `limit` (about 60-150 lines). Read a whole file only when it is short or you need all of it.
- Never re-read a range you have already seen: it is still in the conversation above.
- Batch independent lookups: several greps, globs or reads that do not depend on each other go out as parallel tool calls in one step.
- Keep searches out of dependencies and build output: give grep a `path` or `glob` that excludes `node_modules`, `dist`, `build` output and lock files.
- Stop exploring once you can answer. Answer with repo-relative file paths, function names and line numbers you actually saw, and say what you did not verify.
