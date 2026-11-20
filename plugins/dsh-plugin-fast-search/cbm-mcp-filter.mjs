// cbm-mcp-filter: stdio MCP proxy in front of `codebase-memory-mcp` (CBM) that makes graph answers smaller.
// Wired by config/dsh/cbm.cordis.patch.yml (DSH_CBM=1): command node, args [this file, <cbm binary>, <cbm args>...].
//
// Why (2026-11-14 CBM A/B): the graph cut steps 40% on cross-repo traces but made single lookups slower
// (spi-02 61 -> 138 s, spi-05 136 -> 214 s) because each graph result (~1.6K chars per call) grew the context:
//   - search_graph returns 50 rows and trace_path 100 when the model passes no limit;
//   - 20,270 of 27,028 nodes in one repo are JSON-schema `Variable` rows (pm2/ops JSON files) that crowd out code;
//   - every qualified name starts with the project's encoded path ("Users-abdul.rakib-WebstormProjects-<repo>.").
// What it does:
//   (a) tools/call search_graph, search_code, trace_path without `limit` get limit: CBM_FILTER_DEFAULT_LIMIT (10);
//   (b) search_graph / search_code results drop rows whose label is Variable and whose file ends in .json;
//   (c) result text shortens "<encoded root>-<repo>" to "@<repo>" (root: CBM_FILTER_SHORT_ROOT, default
//       CBM_ALLOWED_ROOT); on the way in, project / qualified_name / function_name / qn_pattern arguments that start
//       with "@<repo>" are expanded back, so names the model copies from a result still resolve.
// Messages are newline-delimited JSON-RPC 2.0; everything else passes through unchanged. A reply that cannot be
// parsed or filtered is forwarded as it came.
//   argv: <codebase-memory-mcp binary> [args...]
import { spawn } from 'node:child_process'
import { createInterface } from 'node:readline'
import { pathToFileURL } from 'node:url'

export const LIMITED_TOOLS = new Set(['search_graph', 'search_code', 'trace_path'])
export const NOISE_FILTERED_TOOLS = new Set(['search_graph', 'search_code'])
const NAME_ARGUMENTS = ['project', 'qualified_name', 'function_name', 'qn_pattern']

// "/Users/abdul.rakib/WebstormProjects" -> "Users-abdul.rakib-WebstormProjects-" (CBM's project-name encoding).
export function encodedRootPrefix(rootPath) {
  const trimmed = String(rootPath ?? '').replace(/^\/+|\/+$/g, '')
  return trimmed ? `${trimmed.replace(/\//g, '-')}-` : ''
}

const escapeRegExp = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

export function shortenProjectNames(text, rootPrefix) {
  if (!rootPrefix) return text
  return text.replace(new RegExp(`${escapeRegExp(rootPrefix)}([A-Za-z0-9_-]+)`, 'g'), '@$1')
}

export function expandProjectName(value, rootPrefix) {
  if (!rootPrefix || typeof value !== 'string') return value
  return value.replace(/^@([A-Za-z0-9_-]+)(?=\.|$)/, `${rootPrefix}$1`)
}

// Arguments of one tools/call after the default limit and the name expansion.
export function rewriteCallArguments(toolName, args, { rootPrefix, defaultLimit }) {
  const rewritten = { ...(args ?? {}) }
  if (LIMITED_TOOLS.has(toolName) && defaultLimit > 0 && rewritten.limit === undefined) rewritten.limit = defaultLimit
  for (const key of NAME_ARGUMENTS) {
    if (typeof rewritten[key] === 'string') rewritten[key] = expandProjectName(rewritten[key], rootPrefix)
  }
  return rewritten
}

const isJsonSchemaNoise = (label, file) => label === 'Variable' && /\.json$/i.test(String(file ?? ''))

// Tree format: a "<section>: N  (cols: qn label file ...)" header followed by indented rows. Rows whose label column
// is Variable and whose file column ends in .json are dropped, and the header's count says how many were hidden.
export function dropJsonSchemaNoiseFromTree(text) {
  const lines = text.split('\n')
  const kept = []
  let columns = null
  let headerIndex = -1
  let hiddenCount = 0
  const closeSection = () => {
    if (headerIndex >= 0 && hiddenCount > 0) {
      kept[headerIndex] = kept[headerIndex].replace(/^(\s*[\w ]+:\s*)(\d+)/, (_, head, count) =>
        `${head}${Number(count) - hiddenCount} (${hiddenCount} JSON-schema Variable rows hidden)`)
    }
    columns = null
    headerIndex = -1
    hiddenCount = 0
  }
  for (const line of lines) {
    const header = line.match(/^\s*[\w ]+:\s*\d+\s+\(cols:\s*([^)]*)\)/)
    if (header) {
      closeSection()
      columns = header[1].trim().split(/\s+/)
      headerIndex = kept.length
      kept.push(line)
      continue
    }
    if (columns && /^\s{2,}\S/.test(line)) {
      const cells = line.trim().split(/\s+/)
      const labelIndex = columns.indexOf('label')
      const fileIndex = columns.indexOf('file')
      if (labelIndex >= 0 && fileIndex >= 0 && isJsonSchemaNoise(cells[labelIndex], cells[fileIndex])) {
        hiddenCount += 1
        continue
      }
      kept.push(line)
      continue
    }
    if (columns && !/^\s/.test(line)) closeSection()
    kept.push(line)
  }
  closeSection()
  return kept.join('\n')
}

// JSON format (format: "json"): drop such rows from every array of row objects.
export function dropJsonSchemaNoiseFromJson(value) {
  if (Array.isArray(value)) {
    return value
      .filter((row) => !(row && typeof row === 'object' && isJsonSchemaNoise(row.label, row.file ?? row.file_path)))
      .map(dropJsonSchemaNoiseFromJson)
  }
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([key, inner]) => [key, dropJsonSchemaNoiseFromJson(inner)]))
  }
  return value
}

export function filterResultText(toolName, text, rootPrefix) {
  let filtered = text
  if (NOISE_FILTERED_TOOLS.has(toolName)) {
    const trimmed = filtered.trimStart()
    if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
      try {
        filtered = JSON.stringify(dropJsonSchemaNoiseFromJson(JSON.parse(filtered)))
      } catch {
        filtered = dropJsonSchemaNoiseFromTree(filtered)
      }
    } else {
      filtered = dropJsonSchemaNoiseFromTree(filtered)
    }
  }
  return shortenProjectNames(filtered, rootPrefix)
}

// A tools/call result after filtering: text parts filtered; structuredContent (if any) filtered the same way.
export function filterCallResult(toolName, result, rootPrefix) {
  if (!result || !Array.isArray(result.content)) return result
  const filtered = {
    ...result,
    content: result.content.map((part) => (part?.type === 'text' && typeof part.text === 'string'
      ? { ...part, text: filterResultText(toolName, part.text, rootPrefix) }
      : part)),
  }
  if (result.structuredContent !== undefined) {
    const structured = NOISE_FILTERED_TOOLS.has(toolName)
      ? dropJsonSchemaNoiseFromJson(result.structuredContent) : result.structuredContent
    filtered.structuredContent = JSON.parse(shortenProjectNames(JSON.stringify(structured), rootPrefix))
  }
  return filtered
}

function runProxy() {
  const [binary, ...binaryArgs] = process.argv.slice(2)
  if (!binary) {
    process.stderr.write('usage: cbm-mcp-filter.mjs <codebase-memory-mcp binary> [args...]\n')
    process.exit(2)
  }
  const options = {
    rootPrefix: encodedRootPrefix(process.env.CBM_FILTER_SHORT_ROOT ?? process.env.CBM_ALLOWED_ROOT ?? ''),
    defaultLimit: Number(process.env.CBM_FILTER_DEFAULT_LIMIT ?? 10),
  }
  const child = spawn(binary, binaryArgs, { stdio: ['pipe', 'pipe', 'inherit'] })
  const pendingCallTools = new Map() // JSON-RPC id -> tool name of a tools/call in flight

  createInterface({ input: process.stdin }).on('line', (line) => {
    if (!line.trim()) return
    let message
    try {
      message = JSON.parse(line)
    } catch {
      child.stdin.write(`${line}\n`)
      return
    }
    if (message.method === 'tools/call' && typeof message.params?.name === 'string') {
      const toolName = message.params.name
      if (message.id !== undefined) pendingCallTools.set(message.id, toolName)
      message = { ...message, params: { ...message.params,
        arguments: rewriteCallArguments(toolName, message.params.arguments, options) } }
      child.stdin.write(`${JSON.stringify(message)}\n`)
      return
    }
    child.stdin.write(`${line}\n`)
  })

  createInterface({ input: child.stdout }).on('line', (line) => {
    if (!line.trim()) return
    let message
    try {
      message = JSON.parse(line)
    } catch {
      process.stdout.write(`${line}\n`)
      return
    }
    if (message.id !== undefined && pendingCallTools.has(message.id)) {
      const toolName = pendingCallTools.get(message.id)
      pendingCallTools.delete(message.id)
      if (message.result) {
        try {
          message = { ...message, result: filterCallResult(toolName, message.result, options.rootPrefix) }
          process.stdout.write(`${JSON.stringify(message)}\n`)
          return
        } catch (error) {
          process.stderr.write(`cbm-mcp-filter: result of ${toolName} passed unfiltered: ${error.message}\n`)
        }
      }
    }
    process.stdout.write(`${line}\n`)
  })

  process.stdin.on('end', () => child.stdin.end())
  child.on('exit', (code, signal) => process.exit(signal ? 1 : (code ?? 0)))
  for (const signalName of ['SIGINT', 'SIGTERM']) process.on(signalName, () => child.kill(signalName))
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) runProxy()
