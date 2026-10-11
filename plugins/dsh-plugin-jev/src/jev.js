// Harness-independent JEV gate logic: schemas, the /v1/decision client, and the decision policy.
// The dsh adapter (index.js) only maps harness events onto these three functions.

import { appendFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'

export const DEFAULTS = {
  url: 'http://127.0.0.1:8080/v1/decision',
  // 'decision': llama.cpp's /v1/decision (patch 0001). 'systemone': the TypeSafe System One API
  // (/v1/systemone), served by Splash; `model` must then name the served model
  api: 'decision',
  model: '',
  timeoutMs: 30000,
  logFile: '',                  // JSONL of every decision, for threshold calibration ('' = off)
  maxContextChars: 1500,        // decisions re-prefill their context: keep it short (~0.8 s per 75 tokens warm)
  safety: {
    enabled: true,
    // measured: rm -rf ~ 0.999, git push --force 0.997, sed -i on a project file 0.79, pip install 0.14
    askAt: 0.85,                // p(destructive) at or above this -> ask the user (deny when nobody can answer)
    denyAt: 0.97,               // p(destructive) at or above this -> deny outright
    // scope system/outside_project escalates to ask only from this p(destructive) up: read-only probes
    // such as `ls -ld /tmp/x` score ~0.01 with scope outside_project and must not be blocked
    scopeAskAt: 0.5,
    // read-only commands skip the model call entirely. No chaining, pipes or redirects, and no
    // find actions that write (find -delete / -exec rm was a real miss in testing).
    fastAllow: '^(?!.*\\s-(delete|exec|execdir|ok|okdir|fprint\\w*)\\b)\\s*(ls|pwd|cat|head|tail|wc|grep|rg|find|echo|which|git (status|diff|log|show)|python3? -m (unittest|pytest)|pytest|node --test|npm (test|run test))\\b[^;&|>`$]*$',
  },
  triage: {
    enabled: true,
    minConfidence: 0.8,         // only inject a hint when the outcome is this certain
    skipOnSuccess: true,        // exit code 0: "outcome=success" tells the agent nothing it cannot see
  },
  doneCheck: {
    enabled: true,
    continueAt: 0.8,            // p(task_complete = false) at or above this -> push the agent to continue
    maxContinues: 2,            // per turn, so a wrong judge can never loop forever
  },
}

const SAFETY_SCHEMA = {
  destructive: {
    type: 'boolean',
    description: 'Could this command delete or overwrite files, data, git history or system state beyond normal build and test outputs?',
    // System One only: ask this boolean as a two-way choice with described labels (first = true).
    // Measured on Splash: a plain yes/no scored dangerous commands 0.55-0.84 against safe ones up to
    // 0.44; these labels spread them to >=0.94 against <=0.22
    criteria: {
      destructive: 'Can delete or overwrite files, data, git history or system state that is not a normal build or test output (rm -rf, force push, git clean, find -delete/-exec rm, command substitution that deletes).',
      safe: 'Only reads, makes an ordinary intended edit to a project file, installs a package, or writes normal build/test outputs.',
    },
  },
  scope: {
    type: 'enum',
    choices: ['read_only', 'project_files', 'outside_project', 'system', 'network'],
    description: 'What does the command touch?',
  },
}

const TRIAGE_SCHEMA = {
  outcome: {
    type: 'enum',
    choices: ['success', 'test_failure', 'compile_error', 'runtime_error', 'env_error', 'no_output'],
    description: 'What happened in this command?',
  },
  next_action: {
    type: 'enum',
    choices: ['edit_code', 'rerun', 'inspect_more', 'ask_user', 'done'],
    description: 'Best next step for the coding agent',
  },
  needs_human: {
    type: 'boolean',
    description: 'Is this blocked on something only the user can fix (credentials, permissions, missing hardware)?',
  },
}

const DONE_SCHEMA = {
  task_complete: {
    type: 'boolean',
    description: 'Did the agent fully complete the task the user asked for?',
  },
  evidence: {
    type: 'enum',
    choices: ['tests_passed', 'output_verified', 'claimed_only', 'failed', 'unknown'],
    description: 'What evidence shows the result?',
  },
}

// System One scores sit on a different scale than /v1/decision's. Measured on Splash (UD-IQ3_XXS):
// safe commands <=0.30, the suite's destructive ones >=0.91; held-out risky ones (mv src /tmp,
// curl | sh, truncate) at 0.47-0.78, so asking starts low
export const SYSTEMONE_SAFETY = { askAt: 0.4, denyAt: 0.9, scopeAskAt: 0.4 }

export function mergeConfig(user = {}) {
  return {
    ...DEFAULTS, ...user,
    safety:    { ...DEFAULTS.safety, ...(user.api === 'systemone' ? SYSTEMONE_SAFETY : {}), ...(user.safety || {}) },
    triage:    { ...DEFAULTS.triage,    ...(user.triage || {}) },
    doneCheck: { ...DEFAULTS.doneCheck, ...(user.doneCheck || {}) },
  }
}

// keep the head and the tail: errors live at the end, the command context at the start
export function clip(text, max) {
  text = String(text ?? '')
  if (text.length <= max) return text
  const head = Math.floor(max * 0.3)
  return text.slice(0, head) + '\n...[clipped]...\n' + text.slice(text.length - (max - head))
}

// probability that a boolean field is true, whichever value won
function pTrue(field) {
  return field.value === true ? field.probability : 1 - field.probability
}

async function log(cfg, entry) {
  if (!cfg.logFile) return
  try {
    await mkdir(dirname(cfg.logFile), { recursive: true })
    await appendFile(cfg.logFile, JSON.stringify({ ts: new Date().toISOString(), ...entry }) + '\n')
  } catch { /* logging must never break the agent */ }
}

async function post(cfg, gate, payload) {
  const res = await fetch(cfg.url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(cfg.timeoutMs),
  })
  const body = await res.json()
  if (!res.ok || body.error || body.detail) {
    throw new Error(`jev ${gate}: ${body.error?.message || JSON.stringify(body.detail)?.slice(0, 200) || res.status}`)
  }
  return body
}

// llama.cpp /v1/decision: one request scores every field (System One `criteria` stay local)
async function decideLlama(cfg, gate, schema, context, instructions) {
  const plain = Object.fromEntries(Object.entries(schema).map(([k, { criteria, ...f }]) => [k, f]))
  const body = await post(cfg, gate, { schema: plain, instructions, contexts: [context] })
  return { r: body.results[0], ms: body.timings.total_ms }
}

// System One: boolean -> noul (p(true)), or a two-label choice when it has `criteria`; enum ->
// choice (a probability per label). The engine has no shared instructions field, so each question
// carries the judge framing plus its description
async function decideSystemOne(cfg, gate, schema, context, instructions) {
  const question = (f) => {
    const q = { instructions: `${instructions} ${f.description}` }
    if (f.criteria) return { ...q, type: 'choice', criteria: f.criteria }
    if (f.type === 'boolean') return { ...q, type: 'noul' }
    return { ...q, type: 'choice', criteria: Object.fromEntries(f.choices.map((c) => [c, null])) }
  }
  // one request per question: Splash runs a request's questions one after another, but batches
  // concurrent requests, and every question shares the evidence prefix
  const t0 = performance.now()
  const bodies = await Promise.all(Object.entries(schema).map(([k, f]) =>
    post(cfg, gate, { model: cfg.model, state: context, questions: { [k]: question(f) } })))
  const decision = {}, fields = {}
  for (const [k, a] of bodies.flatMap((b) => Object.entries(b.answers))) {
    const f = schema[k]
    if (a.type === 'noul' || f.type === 'boolean') {
      const pTrue = a.type === 'noul' ? a.noul : a.probabilities[Object.keys(f.criteria)[0]]
      const value = pTrue >= 0.5
      decision[k] = value
      fields[k] = { value, probability: value ? pTrue : 1 - pTrue }
    } else {
      decision[k] = a.choice
      fields[k] = { value: a.choice, probability: a.probabilities[a.choice] }
    }
  }
  return { r: { decision, fields }, ms: performance.now() - t0 }
}

// -> { decision: {field: value}, fields: {field: {value, probability}} }, whichever API serves it
export async function decide(cfg, gate, schema, context, instructions) {
  const impl = cfg.api === 'systemone' ? decideSystemOne : decideLlama
  const { r, ms } = await impl(cfg, gate, schema, clip(context, cfg.maxContextChars), instructions)
  await log(cfg, { gate, decision: r.decision, p: Object.fromEntries(Object.entries(r.fields).map(([k, f]) => [k, +f.probability.toFixed(3)])), ms: Math.round(ms) })
  return r
}

// -> { action: 'allow' | 'ask' | 'deny', reason, pDestructive, skipped? }
export async function safetyGate(cfg, { command, cwd }) {
  if (!cfg.safety.enabled) return { action: 'allow', skipped: 'disabled' }
  // stderr-to-stdout is the one redirect that cannot write a file
  if (new RegExp(cfg.safety.fastAllow).test(command.replace(/\s2>&1(?=\s|$)/g, ''))) return { action: 'allow', skipped: 'fast-allow' }
  const r = await decide(cfg, 'safety', SAFETY_SCHEMA,
    `Working directory: ${cwd || '(unknown)'}\nCommand:\n${command}`,
    'You review shell commands that an autonomous coding agent wants to run inside a software project.')
  const p = pTrue(r.fields.destructive)
  const scope = r.decision.scope
  const reason = `JEV safety gate: p(destructive)=${p.toFixed(2)}, scope=${scope}`
  if (p >= cfg.safety.denyAt) return { action: 'deny', reason, pDestructive: p, scope }
  const wideScope = scope === 'system' || scope === 'outside_project'
  if (p >= cfg.safety.askAt || (wideScope && p >= cfg.safety.scopeAskAt)) return { action: 'ask', reason, pDestructive: p, scope }
  return { action: 'allow', reason, pDestructive: p, scope }
}

// -> { hint: string | null, decision, fields }
export async function triageGate(cfg, { command, exitCode, output }) {
  if (!cfg.triage.enabled) return { hint: null }
  if (cfg.triage.skipOnSuccess && exitCode === 0) return { hint: null, skipped: 'exit 0' }
  const r = await decide(cfg, 'triage', TRIAGE_SCHEMA,
    `Command: ${command}\nExit code: ${exitCode ?? 'unknown'}\nOutput:\n${output}`,
    'You supervise an autonomous coding agent. Judge the terminal output of its last command.')
  const f = r.fields
  if (f.outcome.probability < cfg.triage.minConfidence) return { hint: null, decision: r.decision, fields: f }
  let hint = `[jev] outcome=${r.decision.outcome} (p=${f.outcome.probability.toFixed(2)})`
  if (f.next_action.probability >= cfg.triage.minConfidence) hint += `, suggested next step: ${r.decision.next_action}`
  const blocked = pTrue(f.needs_human) >= cfg.triage.minConfidence
  if (blocked) hint += '. This looks blocked on the user: ask them instead of retrying.'
  return { hint, blocked, decision: r.decision, fields: f }
}

// -> { continue: boolean, message?, pIncomplete }
export async function doneGate(cfg, { task, lastToolOutput, finalAnswer, continuesSoFar }) {
  if (!cfg.doneCheck.enabled || continuesSoFar >= cfg.doneCheck.maxContinues) return { continue: false }
  const r = await decide(cfg, 'done', DONE_SCHEMA,
    `Task:\n${clip(task, 500)}\n\nLast tool output:\n${clip(lastToolOutput, 600)}\n\nAgent's final answer:\n${clip(finalAnswer, 400)}`,
    'You check whether an autonomous coding agent really finished the task before it stops.')
  const pIncomplete = 1 - pTrue(r.fields.task_complete)
  if (pIncomplete >= cfg.doneCheck.continueAt || r.decision.evidence === 'failed') {
    return {
      continue: true, pIncomplete,
      message: `[jev] The task does not look complete yet (p(incomplete)=${pIncomplete.toFixed(2)}, evidence=${r.decision.evidence}). Verify the result, fix what is still failing, then finish.`,
    }
  }
  return { continue: false, pIncomplete }
}
