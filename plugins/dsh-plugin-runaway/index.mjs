// dsh-plugin-runaway: a reply that hits max_tokens ends the whole turn (dsh-agent-loop step(): finish "max-tokens"
// returns before any tool call runs), and headless exits. With reasoning at xhigh the model sometimes deliberates past
// the 16,384-token cap without writing anything (eval 2026-11-02: t2-semver, t4-toposort and t6-mdtoc, 0 hidden tests,
// the starter stub untouched). When the last reply was cut that way, steer the agent back instead of letting the turn
// close, and send that one next request with thinking off (reasoningEffort "off", which the Splash patch maps to
// reasoning_effort "none"): a text nudge alone was ignored, the model thought to the cap twice more (t2, 2026-11-02).
// The step after runs at the configured effort again. At most maxContinues per turn: each cut reply stays in history.
// A rescued headless run still exits 1: the turn keeps "max-tokens" as its end reason once any step hit it
// (dsh-agent-loop turn(), dsh-headless exit). When the guard continued the run and the last reply ended normally
// ("stop"), the exit code is set back to 0 from process's exit event.
import Schema from '@deepseek-ai/schemastery'
import { createUserMessage } from '@deepseek-ai/dsh-llm'

export const name = 'runaway'

export const Config = Schema.object({
  maxContinues: Schema.number().default(2),
  thinkingOff: Schema.boolean().default(true),
  message: Schema.string().default(''),
  verbose: Schema.boolean().default(true),
})

const SOURCE = { kind: 'plugin', plugin: 'runaway' }
const MESSAGE = 'Your previous reply hit the output limit while you were still thinking, so nothing was done. ' +
  'Thinking is off for this step: act now on the plan from your reasoning above. Make the change with the file ' +
  'tools, then run the tests; you can refine after you see the results.'

export function apply(ctx, config) {
  const maxContinues = config.maxContinues ?? 2
  const thinkingOff = config.thinkingOff ?? true
  const say = (msg) => { if (config.verbose !== false) process.stderr.write(`[runaway] ${msg}\n`) }
  const state = new WeakMap()
  let rescued = 0
  let lastFinish = ''
  process.on('exit', (code) => {
    if (code === 1 && rescued > 0 && lastFinish === 'stop') process.exitCode = 0
  })
  const stateOf = (agent) => {
    if (!state.has(agent)) state.set(agent, { turn: -1, continues: 0, cut: false, noThink: false })
    return state.get(agent)
  }

  ctx.on('agent/pre-step', async (payload, next) => {
    const s = stateOf(payload.agent)
    if (s.turn !== payload.turn) Object.assign(s, { turn: payload.turn, continues: 0 })
    s.cut = false
    return next()
  })

  ctx.on('agent/request', async (payload, next) => {
    const cfg = await next()
    const s = stateOf(payload.agent)
    if (!s.noThink) return cfg
    s.noThink = false
    say('this request runs with thinking off')
    return { ...cfg, reasoningEffort: 'off' }
  })

  ctx.on('agent/assistant-stream', ({ agent, frame }) => {
    if (frame.type !== 'chunk' || frame.chunk?.type !== 'finish') return
    lastFinish = frame.chunk.reason?.kind ?? ''
    stateOf(agent).cut = lastFinish === 'max-tokens'
  })

  ctx.on('agent/turn-stopping', async ({ agent }) => {
    const s = stateOf(agent)
    if (!s.cut) return
    if (s.continues >= maxContinues) { say(`reply cut by max_tokens; ${s.continues} continues used, letting the turn end`); return }
    s.continues += 1
    rescued += 1
    s.cut = false
    s.noThink = thinkingOff
    say(`reply cut by max_tokens while thinking -> continue (${s.continues}/${maxContinues})`)
    agent.steer(createUserMessage({ content: [{ type: 'text', text: config.message || MESSAGE }], source: SOURCE }))
  })
}
