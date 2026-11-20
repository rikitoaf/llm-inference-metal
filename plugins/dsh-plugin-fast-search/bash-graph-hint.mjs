// dsh-plugin-bash-graph-hint: dsh's own bash tool (@deepseek-ai/dsh-tool-bash), unchanged, plus a one-line
// "graph first" hint under the result of a code-navigation command (grep -r/rg, find, ls, cat, head, tail, sed -n)
// while the codebase-memory-mcp tools are on. Replaces the `tool-bash` row (same config).
//
// Why: after glob and grep got hints, the 3-bit model went around them and navigated with bash
// (`grep -rn`, `sed -n`, `ls`: 38 of 41 navigation calls in the 2026-11-11 session).
import { apply as applyBash, Config as BashConfig, inject as bashInject } from '@deepseek-ai/dsh-tool-bash'

export const name = 'bash-graph-hint'
export const inject = bashInject
export const Config = BashConfig

// Any grep/rg/ag/ack that searches code: recursive (-r, -R, -rni, ...) or naming a code file.
const SEARCH_COMMAND = /(^|[;&|]\s*)(grep|egrep|rg|ag|ack)\s/
const FIND_COMMAND = /(^|[;&|]\s*)(find|ls|tree)\s/
const READ_COMMAND = /(^|[;&|]\s*)(sed\s+-n|cat|head|tail)\s/
const CODE_FILE = /\.(js|mjs|cjs|ts|tsx|jsx|vue|py|go|java|kt|rb|php|cs|rs|c|cc|cpp|h|hpp|swift|scala|sql)\b/

function graphAvailable() { return process.env.DSH_CBM === '1' }

// The search term of a `grep -rn "term" ...` command, when it is a plain word.
function searchTerm(command) {
  const quoted = command.match(/(?:grep|rg|ag|ack)\s+(?:-\S+\s+)*["']([^"']+)["']/)
  const bare = command.match(/(?:grep|rg|ag|ack)\s+(?:-\S+\s+)*([A-Za-z_$][\w$|\\]*)/)
  return (quoted?.[1] ?? bare?.[1] ?? '').replace(/\\\|/g, '|')
}

// The hint for one bash command, or '' when the command is not code navigation.
export function bashGraphHint(command) {
  const text = String(command ?? '')
  if (SEARCH_COMMAND.test(text) && (/\s-[a-zA-Z]*[rR]/.test(text) || /(^|[;&|]\s*)(rg|ag|ack)\s/.test(text) || CODE_FILE.test(text) || /--include/.test(text))) {
    const term = searchTerm(text)
    const pattern = term ? `"${term}"` : '"<text>"'
    return `\n\n[graph first] Next time use mcp__cbm__search_code({project, pattern: ${pattern}}): same hits, each ` +
      'mapped to its function, from the index; for a symbol name use mcp__cbm__search_graph({project, name_pattern}). ' +
      'Shell grep is the fallback for non-code files and fresh edits.'
  }
  if (READ_COMMAND.test(text) && CODE_FILE.test(text)) {
    return '\n\n[graph first] To read code use mcp__cbm__get_file_outline({project, file_path}) then ' +
      'mcp__cbm__get_code_snippet({project, qualified_name}) for the one function you need, instead of sed/cat ranges.'
  }
  if (FIND_COMMAND.test(text)) {
    return '\n\n[graph first] To find files or symbols use mcp__cbm__search_graph({project, file_pattern or ' +
      'name_pattern}) or get_architecture once per repo; ls/find are the fallback.'
  }
  return ''
}

export function apply(ctx, config) {
  const hintingTools = new Proxy(ctx.tools, {
    get(target, property) {
      if (property !== 'register') {
        const value = Reflect.get(target, property, target)
        return typeof value === 'function' ? value.bind(target) : value
      }
      return (tool) => {
        const render = tool?.output?.render
        if (tool?.name !== 'bash' || typeof render !== 'function') return target.register(tool)
        return target.register({
          ...tool,
          output: {
            ...tool.output,
            render: (args, value) => {
              const parts = render(args, value)
              const hint = graphAvailable() ? bashGraphHint(args?.command) : ''
              return hint && Array.isArray(parts) ? [...parts, { type: 'text', text: hint }] : parts
            },
          },
        })
      }
    },
  })
  const hintingContext = new Proxy(ctx, {
    get(target, property) {
      if (property === 'tools') return hintingTools
      const value = Reflect.get(target, property, target)
      return typeof value === 'function' ? value.bind(target) : value
    },
  })
  return applyBash(hintingContext, config)
}
