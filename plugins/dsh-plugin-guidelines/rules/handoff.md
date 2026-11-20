# Session handoff

When the user asks for a handoff (or runs /handoff), write `{{handoffFile}}` in the working directory so a new
session, or another agent, can continue without this conversation. Overwrite an older one. Use these sections and
skip any that would be empty:

```
# Handoff: <task title>
Written: <ISO date> · Branch: <git branch> · Status: <in progress | blocked | done>

## Goal                      what the user wants, in one or two sentences
## Completed                 what is done and verified, with file paths
## Not yet done              remaining steps, in order
## Failed approaches         what was tried and why it failed, with the exact error message (do not repeat these)
## Key decisions             choices made and the reason for each
## Current state             uncommitted changes, running processes, test status
## Code context              the files and functions that matter, path:line
## Resume instructions       the first command or step the next session should take
## Warnings                  traps, flaky tests, things that look wrong but are intended
```

Be concrete: exact commands, paths and error text, not descriptions of them. A quick handoff keeps only Goal,
Not yet done, Current state and Resume instructions.

When resuming from a handoff: read it, check `git status` and `git log -3` for drift since it was written, say in
two lines what you will continue with, then continue from Resume instructions.
