// Unit test for plugins/dsh-plugin-purpose-effort: drives its llm/stream listener through a cordis-style waterfall
// (one shared options object, next() without arguments) and checks which requests change. Run:
//   node scripts/test-purpose-effort.mjs        exits non-zero on any mismatch
import { apply } from '../plugins/dsh-plugin-purpose-effort/index.mjs'

function harness(config) {
  const listeners = []
  const ctx = { on: (event, cb, opts) => { if (event === 'llm/stream') listeners.push({ cb, opts }) } }
  apply(ctx, config)
  // mirrors cordis Events.waterfall: listeners in order, then the adapter, all with the same args
  const stream = (options) => {
    let i = 0
    const next = () => (i < listeners.length ? listeners[i++].cb(options, next) : { sent: { ...options } })
    return next()
  }
  return { listeners, stream }
}

let failures = 0
const check = (label, got, want) => {
  const ok = JSON.stringify(got) === JSON.stringify(want)
  if (!ok) failures++
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${JSON.stringify(got)}${ok ? '' : ` (want ${JSON.stringify(want)})`}`)
}

const quiet = { verbose: false }
{
  const { listeners, stream } = harness(quiet)
  check('registers one global llm/stream listener', listeners.map((l) => l.opts), [{ global: true }])
  check('compaction request -> off', stream({ purpose: 'compaction', maxTokens: 8192 }).sent.reasoningEffort, 'off')
  check('compaction keeps other options', stream({ purpose: 'compaction', maxTokens: 8192, sessionId: 's1' }).sent,
    { purpose: 'compaction', maxTokens: 8192, sessionId: 's1', reasoningEffort: 'off' })
  check('agent step (no purpose) untouched', stream({ messages: [], reasoningEffort: 'high' }).sent.reasoningEffort, 'high')
  check('agent step without effort stays default', 'reasoningEffort' in stream({ messages: [] }).sent, false)
  check('other purpose untouched', stream({ purpose: 'title', reasoningEffort: 'high' }).sent.reasoningEffort, 'high')
}
{
  const { stream } = harness({ ...quiet, efforts: { compaction: 'low', title: 'off' } })
  check('configured efforts: compaction -> low', stream({ purpose: 'compaction' }).sent.reasoningEffort, 'low')
  check('configured efforts: title -> off', stream({ purpose: 'title', reasoningEffort: 'high' }).sent.reasoningEffort, 'off')
}
{
  const { stream } = harness({ ...quiet, efforts: {} })
  check('empty efforts: compaction untouched', 'reasoningEffort' in stream({ purpose: 'compaction' }).sent, false)
}

console.log(failures ? `\n${failures} failure(s)` : '\nall passed')
process.exit(failures ? 1 : 0)
