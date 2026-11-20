// dsh-plugin-purpose-effort: set the reasoning effort of non-agent model calls by their `purpose`.
// dsh-compaction-basic summarizes with a plain ctx.llm.stream({ purpose: 'compaction', maxTokens: 8192 }) and no
// reasoningEffort, so the provider default (high, which the Qwen template maps to xhigh) applies. In real sessions
// (2026-10-09 .. 11-03) the five compactions took 230-465 s each, and one ended after 261 s with "summarization
// truncated at the token cap": the hidden reasoning used the 8,192-token budget before the summary was written.
// This listener rewrites only the requests whose purpose is listed (default: compaction -> off, which the Splash
// patch maps to reasoning_effort "none", as the runaway guard does). The agent's own steps carry no purpose and
// keep their configured effort. The llm/stream waterfall passes one options object to every listener and the
// adapter, so the change is made in place before next().
import Schema from '@deepseek-ai/schemastery'

export const name = 'purpose-effort'

export const Config = Schema.object({
  efforts: Schema.dict(Schema.string()).default({ compaction: 'off' }),
  verbose: Schema.boolean().default(true),
})

export function apply(ctx, config) {
  const efforts = config.efforts ?? { compaction: 'off' }
  const say = (msg) => { if (config.verbose !== false) process.stderr.write(`[purpose-effort] ${msg}\n`) }
  ctx.on('llm/stream', (options, next) => {
    const effort = options.purpose === undefined ? undefined : efforts[options.purpose]
    if (effort !== undefined && options.reasoningEffort !== effort) {
      say(`${options.purpose} request runs with reasoningEffort ${effort}`)
      options.reasoningEffort = effort
    }
    return next()
  }, { global: true })
}
