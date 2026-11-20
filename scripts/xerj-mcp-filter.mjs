// xerj-mcp-filter: stdio MCP proxy in front of `xerj mcp` that exposes only an allowlist of its tools.
// dsh-mcp-client has no per-tool filter, and `xerj mcp` publishes 22 tools (~55K chars of schema, prefilled
// on every request). The allowlist below is ~23K chars and leaves out xerj_batch_search (5+ GB of node RAM
// over a day, ~196K-char replies) and xerj_field_report (publishes to GitHub / xerj.org).
//   argv: <xerj binary> [xerj mcp args...]     env: XERJ_MCP_TOOLS=comma,list overrides the allowlist
// Messages are newline-delimited JSON-RPC 2.0. tools/list replies are filtered; a tools/call for a hidden
// tool is answered here with an error and never reaches xerj. Everything else passes through unchanged.
// When the node at XERJ_URL does not answer at tools/list time, the list is empty: `xerj mcp` lists its tools
// without a node, so the agent would otherwise pay their schemas on every request and spend a step on "no XERJ
// node is listening" before falling back to grep. dsh lists tools at startup: start the node, then dsh.
import { spawn } from 'node:child_process'
import { createInterface } from 'node:readline'

const DEFAULT_TOOLS = ['xerj_def', 'xerj_search', 'xerj_semantic_search', 'xerj_hybrid_search', 'xerj_map']
const allowed = new Set(
  (process.env.XERJ_MCP_TOOLS || DEFAULT_TOOLS.join(',')).split(',').map((s) => s.trim()).filter(Boolean),
)
const [bin, ...args] = process.argv.slice(2)
if (!bin) {
  process.stderr.write('usage: xerj-mcp-filter.mjs <xerj binary> mcp [args...]\n')
  process.exit(2)
}

const child = spawn(bin, args, { stdio: ['pipe', 'pipe', 'inherit'] })
const pendingLists = new Set() // ids of tools/list requests whose replies need filtering

const send = (stream, msg) => stream.write(JSON.stringify(msg) + '\n')

const nodeUrl = process.env.XERJ_URL || 'http://localhost:9200'
const nodeUp = () => fetch(nodeUrl, { signal: AbortSignal.timeout(1500) }).then((r) => r.ok, () => false)

createInterface({ input: process.stdin }).on('line', (line) => {
  if (!line.trim()) return
  let msg
  try {
    msg = JSON.parse(line)
  } catch {
    child.stdin.write(line + '\n') // not ours to judge; let xerj answer the parse error
    return
  }
  if (msg.method === 'tools/list' && msg.id !== undefined) pendingLists.add(msg.id)
  if (msg.method === 'tools/call' && !allowed.has(msg.params?.name)) {
    if (msg.id !== undefined) {
      send(process.stdout, {
        jsonrpc: '2.0',
        id: msg.id,
        result: {
          isError: true,
          content: [{ type: 'text', text: `${msg.params?.name} is not enabled. Available: ${[...allowed].join(', ')}` }],
        },
      })
    }
    return
  }
  child.stdin.write(line + '\n')
})

createInterface({ input: child.stdout }).on('line', (line) => {
  if (!line.trim()) return
  let msg
  try {
    msg = JSON.parse(line)
  } catch {
    process.stdout.write(line + '\n')
    return
  }
  if (msg.id !== undefined && pendingLists.delete(msg.id) && Array.isArray(msg.result?.tools)) {
    nodeUp().then((up) => {
      msg.result.tools = up ? msg.result.tools.filter((t) => allowed.has(t.name)) : []
      if (!up) process.stderr.write(`xerj-mcp-filter: no xerj node at ${nodeUrl}; listing no tools\n`)
      send(process.stdout, msg)
    })
    return
  }
  process.stdout.write(line + '\n')
})

process.stdin.on('end', () => child.stdin.end())
child.on('exit', (code, signal) => process.exit(signal ? 1 : (code ?? 0)))
for (const sig of ['SIGINT', 'SIGTERM']) process.on(sig, () => child.kill(sig))
