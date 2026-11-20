// dsh-plugin-fast-search: dsh's own glob and grep tools (@deepseek-ai/dsh-tool-fs-search), unchanged except that
// ripgrep skips dependency and build directories. Replaces the `tool-fs-search` row (same config keys).
//
// Why: the shipped glob runs `rg --files --no-ignore --hidden`, so from a multi-repo root it walks every file,
// node_modules included. On ~/WebstormProjects that is 328K files: 5.6-8.4 s per glob call (2026-11-11 session).
// With the excludes below the same glob takes 0.22 s and a root-wide grep 0.24 s. Grep already honours .gitignore
// inside git repos, but non-git folders and vendored trees were searched in full.
//
// How: the tools are registered by the package's own applyGlobTool/applyGrepTool against a context whose
// `subprocess.spawn` appends `--glob=!**/<dir>` excludes to every ripgrep argv. A search whose `path` is inside an
// excluded directory still works: those excludes are skipped when the requested path names that directory.
import Schema from '@deepseek-ai/schemastery'
import {
  applyGlobTool, applyGrepTool, inject as searchInject, GLOB_MAX_RESULTS, GREP_MAX_MATCHES, GREP_MAX_LINE_BYTES,
  RAW_OUTPUT_MAX_BYTES, SEARCH_GRACE_MS, SEARCH_META_MAX_BYTES, SEARCH_STDERR_MAX_BYTES, SEARCH_TIMEOUT_MS,
} from '@deepseek-ai/dsh-tool-fs-search'

export const name = 'fast-search'
export const inject = searchInject

const DEFAULT_EXCLUDED_DIRECTORIES = [
  'node_modules', 'dist', 'build', '.next', '.nuxt', 'coverage', 'vendor', '.cache', '.turbo', '.angular', 'target',
]

export const Config = Schema.object({
  sampleOverCapGlobResults: Schema.boolean().default(false),
  globMaxResults: Schema.number().default(GLOB_MAX_RESULTS),
  grepMaxMatches: Schema.number().default(GREP_MAX_MATCHES),
  grepMaxLineBytes: Schema.number().default(GREP_MAX_LINE_BYTES),
  searchMetaMaxBytes: Schema.number().default(SEARCH_META_MAX_BYTES),
  rawOutputMaxBytes: Schema.number().default(RAW_OUTPUT_MAX_BYTES),
  graceMs: Schema.number().default(SEARCH_GRACE_MS),
  stderrMaxBytes: Schema.number().default(SEARCH_STDERR_MAX_BYTES),
  timeoutMs: Schema.number().default(SEARCH_TIMEOUT_MS),
  excludedDirectories: Schema.array(Schema.string()).default(DEFAULT_EXCLUDED_DIRECTORIES),
})

// The excludes for one ripgrep argv: skip any directory the caller explicitly searches in (its path names it).
export function excludeArguments(argv, excludedDirectories) {
  const separatorIndex = argv.lastIndexOf('--')
  const searchPath = separatorIndex >= 0 ? String(argv[separatorIndex + 1] ?? '') : ''
  const pathSegments = new Set(searchPath.split(/[\\/]/))
  return excludedDirectories
    .filter((directory) => !pathSegments.has(directory))
    .flatMap((directory) => [`--glob=!**/${directory}`, `--glob=!**/${directory}/**`])
}

// Insert the excludes before `--` (the search root) or at the end.
export function withExcludes(argv, excludedDirectories) {
  const excludes = excludeArguments(argv, excludedDirectories)
  if (excludes.length === 0) return argv
  const separatorIndex = argv.lastIndexOf('--')
  if (separatorIndex < 0) return [...argv, ...excludes]
  return [...argv.slice(0, separatorIndex), ...excludes, ...argv.slice(separatorIndex)]
}

// Graph first (user, 2026-11-11): when the codebase-memory-mcp tools are present, glob and grep are the fallback.
// The 3-bit model ignored a system-prompt rule (0 graph calls of 23 in a live session), so the nudge sits where it
// decides: in glob's and grep's own tool instructions and as one line under each of their results.
const GRAPH_SEARCH_TOOL = 'mcp__cbm__search_graph'
const GRAPH_FIRST_INSTRUCTIONS = {
  'tool:glob': 'Code graph first: when mcp__cbm__ tools are present, find files and symbols with mcp__cbm__search_graph '
    + '(file_pattern or name_pattern, project = the repository folder name or absolute path); it answers from a '
    + 'prebuilt index in milliseconds. Use glob as the fallback: when the graph returns nothing, for files the graph '
    + 'does not index (configs, docs, data), or right after edits the index has not seen yet. ',
  'tool:grep': 'Code graph first: for definitions, callers, callees and request flows use mcp__cbm__search_graph, '
    + 'mcp__cbm__trace_path and mcp__cbm__get_code_snippet. Use grep as the fallback: for string literals, config '
    + 'values, log messages, non-code files, or when a graph query returns nothing. `include` takes ONE glob: '
    + 'write "*.{vue,js}", never a comma-separated list. ',
}
const IDENTIFIER = /^[A-Za-z_$][\w$]{2,}$/

// DSH_CBM=1 (scripts/dsh-env.sh, set by scripts/work) registers the graph tools; MCP tools may be visible only
// within a session scope, so the env flag counts as well.
function graphAvailable(ctx, scope) {
  if (process.env.DSH_CBM === '1') return true
  try { return ctx.tools.get(GRAPH_SEARCH_TOOL, scope) !== undefined } catch { return false }
}

// The one-line nudge under a glob or grep result, or '' where the graph would not have been the better call.
export function graphHint(toolName, args) {
  if (toolName === 'glob') {
    return `\n\n[graph first] Next time try ${GRAPH_SEARCH_TOOL}({project, file_pattern: "${args?.pattern ?? '*'}"}): ` +
      'the code graph answers file lookups in milliseconds; glob is the fallback.'
  }
  if (toolName === 'grep' && IDENTIFIER.test(String(args?.pattern ?? ''))) {
    return `\n\n[graph first] For the symbol "${args.pattern}" try ${GRAPH_SEARCH_TOOL}({project, name_pattern: ` +
      `"${args.pattern}"}) then mcp__cbm__trace_path or get_code_snippet; grep is the fallback for strings and non-code files.`
  }
  return ''
}

// Wrap a glob/grep tool so its rendered result carries the hint when the graph tools are registered.
function withGraphHint(ctx, tool) {
  if (tool.name !== 'glob' && tool.name !== 'grep') return tool
  const render = tool.output?.render
  if (typeof render !== 'function') return tool
  return {
    ...tool,
    output: {
      ...tool.output,
      render: (args, value) => {
        const parts = render(args, value)
        const hint = graphAvailable(ctx) ? graphHint(tool.name, args) : ''
        if (!hint || !Array.isArray(parts)) return parts
        return [...parts, { type: 'text', text: hint }]
      },
    },
  }
}

export function apply(ctx, config) {
  const excludedDirectories = config.excludedDirectories ?? DEFAULT_EXCLUDED_DIRECTORIES
  const filteringSubprocess = new Proxy(ctx.subprocess, {
    get(target, property) {
      if (property === 'spawn') {
        return (options) => target.spawn({ ...options, argv: withExcludes(options.argv, excludedDirectories) })
      }
      const value = Reflect.get(target, property, target)
      return typeof value === 'function' ? value.bind(target) : value
    },
  })
  const graphFirstTools = new Proxy(ctx.tools, {
    get(target, property) {
      if (property === 'register') return (tool) => target.register(withGraphHint(ctx, tool))
      const value = Reflect.get(target, property, target)
      return typeof value === 'function' ? value.bind(target) : value
    },
  })
  const graphFirstSystemPrompt = new Proxy(ctx.systemPrompt, {
    get(target, property) {
      if (property === 'section') {
        return (section) => {
          const prefix = GRAPH_FIRST_INSTRUCTIONS[section?.name]
          if (!prefix || typeof section.text !== 'function') return target.section(section)
          const original = section.text
          return target.section({
            ...section,
            text: (input) => {
              const text = original(input)
              return text && graphAvailable(ctx, input?.scope) ? prefix + text : text
            },
          })
        }
      }
      const value = Reflect.get(target, property, target)
      return typeof value === 'function' ? value.bind(target) : value
    },
  })
  const searchContext = new Proxy(ctx, {
    get(target, property) {
      if (property === 'subprocess') return filteringSubprocess
      if (property === 'tools') return graphFirstTools
      if (property === 'systemPrompt') return graphFirstSystemPrompt
      const value = Reflect.get(target, property, target)
      return typeof value === 'function' ? value.bind(target) : value
    },
  })
  const shared = {
    maxMetaBytes: config.searchMetaMaxBytes,
    rawOutputMaxBytes: config.rawOutputMaxBytes,
    graceMs: config.graceMs,
    stderrMaxBytes: config.stderrMaxBytes,
    timeoutMs: config.timeoutMs,
  }
  applyGlobTool(searchContext, {
    ...shared, sampleOverCapGlobResults: config.sampleOverCapGlobResults, maxResults: config.globMaxResults,
  })
  applyGrepTool(searchContext, { ...shared, maxMatches: config.grepMaxMatches, maxLineBytes: config.grepMaxLineBytes })
}
