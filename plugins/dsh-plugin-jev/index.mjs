// dsh-plugin-jev: JEV gates for DeepSeek Harness, backed by llama-server's /v1/decision.
//   tools/pre-execute   -> safety gate on bash commands (allow / ask / deny)
//   tools/post-execute  -> triage of bash output, injected as a short hint for the model
//   agent/turn-stopping -> done check; steers the agent back to work when the task looks unfinished
// Every gate fails open: if the decision server is down, the agent runs as if the plugin were absent.
import Schema from '@deepseek-ai/schemastery'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { mergeConfig, safetyGate, triageGate, doneGate, clip } from './src/jev.js'

export const name = 'jev'

export const Config = Schema.object({
  url: Schema.string().default('http://127.0.0.1:8080/v1/decision'),
  api: Schema.union(['decision', 'systemone']).default('decision'),
  model: Schema.string().default(''),
  timeoutMs: Schema.number().default(30000),
  logFile: Schema.string().default(''),
  verbose: Schema.boolean().default(true),
  maxContextChars: Schema.number().default(1500),
  safety: Schema.object({
    enabled: Schema.boolean().default(true),
    askAt: Schema.number().default(0.85),
    denyAt: Schema.number().default(0.97),
    scopeAskAt: Schema.number().default(0.5),
    fastAllow: Schema.string(),
  }),
  triage: Schema.object({
    enabled: Schema.boolean().default(true),
    minConfidence: Schema.number().default(0.8),
    skipOnSuccess: Schema.boolean().default(true),
  }),
  doneCheck: Schema.object({
    enabled: Schema.boolean().default(true),
    continueAt: Schema.number().default(0.8),
    maxContinues: Schema.number().default(2),
  }),
})

const SOURCE = { kind: 'plugin', plugin: 'jev' }
const textOf = (blocks = []) => blocks.filter((b) => b?.type === 'text').map((b) => b.text).join('')

// one-shot bash returns a structured value; persistent bash ends its text with an exit-code marker
function exitCodeOf(result) {
  const v = result.value
  if (v && typeof v === 'object' && v.kind === 'foreground') return v.exitCode
  const m = /\[(?:Command finished with exit code|exit code:) (-?\d+)\]/.exec(textOf(result.content))
  return m ? Number(m[1]) : undefined
}

export function apply(ctx, rawConfig) {
  const cfg = mergeConfig(Object.fromEntries(Object.entries(rawConfig).filter(([, v]) => v !== undefined)))
  if (!rawConfig.safety?.fastAllow) cfg.safety.fastAllow = mergeConfig().safety.fastAllow
  const say = (msg) => { if (cfg.verbose) process.stderr.write(`[jev] ${msg}\n`) }
  // per agent: last bash result (for the done check), continues used in the current turn, and whether
  // the turn hit a wall only the user can remove (a gated command, or triage saying "ask the user")
  const state = new WeakMap()
  const stateOf = (agent) => {
    if (!state.has(agent)) state.set(agent, { lastBash: null, turn: -1, continues: 0, blocked: null })
    return state.get(agent)
  }
  const newTurn = (s, turn) => { if (s.turn !== turn) Object.assign(s, { turn, continues: 0, blocked: null }) }

  ctx.on('agent/pre-step', async (payload, next) => {
    newTurn(stateOf(payload.agent), payload.turn)
    return next()
  })

  ctx.on('tools/pre-execute', async (exec, next) => {
    if (exec.name !== 'bash' || typeof exec.arguments?.command !== 'string') return next()
    try {
      const g = await safetyGate(cfg, { command: exec.arguments.command, cwd: exec.agent?.session.header.cwd })
      if (g.skipped) return next()
      say(`safety ${g.action.toUpperCase()} (p_destructive=${g.pDestructive.toFixed(2)}, scope=${g.scope}): ${clip(exec.arguments.command, 120)}`)
      if (g.action !== 'allow' && exec.agent) stateOf(exec.agent).blocked = `safety gate ${g.action}`
      if (g.action === 'deny') return { kind: 'deny', reason: g.reason }
      if (g.action === 'ask') return { kind: 'ask', reason: g.reason }   // headless has no approver: ask fails closed
    } catch (error) {
      say(`safety gate unavailable, allowing: ${error.message}`)
    }
    return next()
  })

  ctx.on('tools/post-execute', async (exec, result, next) => {
    const downstream = await next()
    if (exec.name !== 'bash') return downstream
    const command = exec.arguments?.command ?? ''
    const output = textOf(result.content)
    const exitCode = exitCodeOf(result)
    if (exec.agent) stateOf(exec.agent).lastBash = { command, output, exitCode }
    try {
      const t = await triageGate(cfg, { command, exitCode, output })
      if (!t.hint) return downstream
      if (t.blocked && exec.agent) stateOf(exec.agent).blocked = 'triage: needs the user'
      say(`triage -> ${t.hint.replace(/^\[jev\] /, '')}`)
      const extra = createUserMessage({ content: [{ type: 'text', text: t.hint }], source: SOURCE })
      return { ...downstream, additionalContexts: [extra, ...(downstream.additionalContexts ?? [])] }
    } catch (error) {
      say(`triage gate unavailable: ${error.message}`)
      return downstream
    }
  })

  ctx.on('agent/turn-stopping', async ({ agent, turn }) => {
    const s = stateOf(agent)
    newTurn(s, turn)
    // stopping to report a wall is the correct behaviour; pushing on only breeds workarounds
    if (s.blocked) { say(`done-check skipped: ${s.blocked}`); return }
    let task = '', finalAnswer = ''
    try {
      const messages = agent.session.deriveMessages()
      task = textOf(messages.find((m) => m.role === 'user' && m.source?.kind === 'user')?.content)
      finalAnswer = textOf([...messages].reverse().find((m) => m.role === 'assistant')?.content)
    } catch { /* history unavailable: judge from the last tool output only */ }
    if (!task) return
    try {
      const last = s.lastBash
      const d = await doneGate(cfg, {
        task,
        lastToolOutput: last ? `$ ${last.command}\n[exit ${last.exitCode ?? '?'}]\n${last.output}` : '(no command was run)',
        finalAnswer,
        continuesSoFar: s.continues,
      })
      if (d.pIncomplete !== undefined) say(`done-check: p_incomplete=${d.pIncomplete.toFixed(2)} -> ${d.continue ? 'CONTINUE' : 'stop'}`)
      if (d.continue) {
        s.continues += 1
        agent.steer(createUserMessage({ content: [{ type: 'text', text: d.message }], source: SOURCE }))
      }
    } catch (error) {
      say(`done check unavailable: ${error.message}`)
    }
  })
}
