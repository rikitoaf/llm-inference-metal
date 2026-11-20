// dsh-plugin-graph-first-reminder: short user-side <system-reminder>s. Loaded with the codebase-memory-mcp patch
// (config/dsh/cbm.cordis.patch.yml), so it is active only with DSH_CBM=1. It injects:
// - the turn reminder (batching, graph first, fallback, scope rule) at the start of every user turn;
// - the drift reminder when a long turn drifts back to shell navigation;
// - the scope reminder once, when the first edit/write of a turn follows a request with no action verb
//   (2026-11-18: 3 of 18 aborted turns were edits nobody asked for, 2 of them answering questions);
// - the batch reminder after consecutive steps that each made one read-only lookup (72% of steps make one call).
// It only observes tool results (tools/result cannot block a call) and appends messages at agent/pre-step.
//
// Why: in the 2026-11-09..11 sessions the model used the graph heavily only in turns whose user message asked for
// it ("compact and use mcp codebase memory as much as possible": 28 graph calls in that turn), and almost never
// otherwise (0-2 per turn), even with graph-first rules in the system prompt and hints on tool results. A 3-bit
// model follows what is near the end of the conversation. The reminder sits where the user's own words would.
// Injected messages are stored in the session like the per-repo CLAUDE.md reminders (same `agent/pre-step`
// mechanism as dsh-agent-instructions), so the prompt prefix stays cache-stable.
import Schema from '@deepseek-ai/schemastery'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { bashGraphHint } from './bash-graph-hint.mjs'

export const name = 'graph-first-reminder'

export const Config = Schema.object({
  // Re-remind inside a turn after this many steps that navigated without a single graph call (0 = never).
  driftSteps: Schema.number().default(10),
  // After this many consecutive steps that each made exactly one read-only call, suggest batching (0 = never).
  singleLookupSteps: Schema.number().default(2),
  // At most one batch reminder per this many steps.
  batchReminderSpacing: Schema.number().default(6),
  // One reminder when the first edit/write of a turn follows a request with no action verb.
  isScopeCheckEnabled: Schema.boolean().default(true),
  verbose: Schema.boolean().default(true),
})

const GRAPH_PREFIX = 'mcp__cbm__'
const CODE_FILE = /\.(js|mjs|cjs|ts|tsx|jsx|vue|py|go|java|kt|rb|php|cs|rs|c|cc|cpp|h|hpp|swift|scala|sql)$/

// Only code navigation counts toward drift: running scripts, tests, git or reading .env/config files does not
// (2026-11-13: a DB-debugging session got a drift reminder for `node inspect.js` runs).
export function isCodeNavigation(toolName, args) {
  if (toolName === 'glob' || toolName === 'grep') return true
  if (toolName === 'read') return CODE_FILE.test(String(args?.file_path ?? ''))
  if (toolName === 'bash') return bashGraphHint(args?.command) !== ''
  return false
}

export const TURN_REMINDER = '<system-reminder>\nWork rules for this request.\n' +
  '1. Batch: send every lookup that does not depend on another as parallel tool calls in ONE step (several ' +
  'search_code / search_graph / read calls together). One call per step is the slowest way to work.\n' +
  '2. Code graph first (mcp__cbm__ tools, a prebuilt index, milliseconds): search_code instead of grep -rn; ' +
  'search_graph(query or name_pattern) to find a symbol, then get_code_snippet with the exact qualified_name it ' +
  'returned (never guess one); trace_path for callers and flows; project = the repository folder name. Pass ' +
  'limit: 10 on search_graph and search_code; ask for more only if needed.\n' +
  '3. Fallback is normal, not a failure: when a graph call returns 0 results (route and handler files made of ' +
  'anonymous callbacks, configs, .env, SQL, docs, files edited in the last minutes), use grep, glob or a ranged ' +
  'read for that item and keep using the graph for the next one. For a route or handler file, search_code ' +
  'still works: it returns the matching line numbers, then read just those lines.\n' +
  '4. grep include takes one glob: use "*.{vue,js}", not "*.vue,*.js". Re-read a file before editing it if a ' +
  'command or an earlier edit changed it.\n' +
  '5. Scope: a question or a check/explain/analyse request gets an answer or a short plan, then stop: no edits, ' +
  'writes or state-changing commands. Change only what was asked; propose wider changes, do not make them.\n' +
  '</system-reminder>'

export const DRIFT_REMINDER = '<system-reminder>\nThe last steps navigated code with shell/grep/read and no graph ' +
  'call. Use the mcp__cbm__ tools first (search_code, search_graph, get_code_snippet, trace_path); fall back only ' +
  'when they return nothing.\n</system-reminder>'

export const SCOPE_REMINDER = '<system-reminder>\nScope check: this request asked a question or for analysis, not ' +
  'for a change. Unless it asked you to change files, stop editing: say what you changed, then answer or propose ' +
  'a plan and wait.\n</system-reminder>'

export const BATCH_REMINDER = '<system-reminder>\nThe last steps made one lookup each. Send the next independent ' +
  'lookups (read, grep, search_code, search_graph, get_code_snippet) together as parallel calls in one step.\n' +
  '</system-reminder>'

const EDIT_TOOLS = new Set(['edit', 'write'])
// Requests that ask for a change. "can/could/would you <verb>" is a polite imperative; a sentence that is a question
// ("did you change ...?", "should it be ...?") is not, even when it contains one of the verbs (2026-11-18 abort review:
// 2 of 3 over-reach turns were questions answered with edits).
const ACTION_VERB = /\b(implement|fix|change|add|update|remove|delete|create|write|apply|make|refactor|rename|wire|do it|go ahead|proceed)(s|es|ed|d|ing)?\b/i
const QUESTION_START = /^(did|does|do|is|are|was|were|should|shall|will|has|have|had|why|what|how|where|which|who|when|isn't|doesn't)\b/i
const POLITE_REQUEST = /^(can|could|would|will) you\b/i
// A reply that approves a proposed plan ("yes", "ok go ahead", "lgtm") also allows edits.
const APPROVAL = /^(yes|yeah|yep|ok|okay|sure|go|approved?|lgtm|sounds good|looks good)\b/i

function sentencesOf(text) {
  return String(text ?? '').split(/(?<=[.?!])\s+|\n+/).map((sentence) => sentence.trim()).filter(Boolean)
}

// True when some sentence of the user's text asks for a change: it has an action verb and is not a question
// (a polite "can you fix ...?" counts as a request), or approves a plan.
export function asksForChange(text) {
  return sentencesOf(text).some((sentence) => {
    if (APPROVAL.test(sentence)) return true
    if (!ACTION_VERB.test(sentence)) return false
    if (POLITE_REQUEST.test(sentence)) return true
    return !sentence.endsWith('?') && !QUESTION_START.test(sentence)
  })
}

// Read-only lookups for the batch reminder: read, grep, glob, any graph tool, or bash code navigation.
export function isReadOnlyLookup(toolName, args) {
  if (toolName === 'read' || toolName === 'grep' || toolName === 'glob') return true
  if (String(toolName).startsWith(GRAPH_PREFIX)) return true
  return toolName === 'bash' && isCodeNavigation(toolName, args)
}

function userTextOf(messages) {
  return messages
    .filter((message) => message?.role === 'user' && message?.source?.kind === 'user')
    .flatMap((message) => (Array.isArray(message.content) ? message.content : []))
    .filter((part) => part?.type === 'text')
    .map((part) => part.text)
    .join('\n')
}

function reminderMessage(text) {
  return createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'plugin', plugin: name } })
}

// A step's claimed messages hold a message the user typed (not tool results, reminders or runtime context).
function hasUserRequest(messages) {
  return messages.some((message) => message?.role === 'user' && message?.source?.kind === 'user')
}

export function apply(ctx, config) {
  const driftSteps = config.driftSteps ?? 10
  const singleLookupSteps = config.singleLookupSteps ?? 2
  const batchReminderSpacing = config.batchReminderSpacing ?? 6
  const isScopeCheckEnabled = config.isScopeCheckEnabled !== false
  const say = (text) => { if (config.verbose !== false) process.stderr.write(`[graph-first] ${text}\n`) }
  // Per agent (WeakMaps keyed by the agent object), so subagents keep their own counts.
  const navigationStepsWithoutGraph = new WeakMap()
  const stepUsedGraph = new WeakMap()
  const stepUsedNavigation = new WeakMap()
  const stepCalls = new WeakMap()            // calls of the step in progress: { count, readOnlyCount }
  const singleLookupStepRun = new WeakMap()  // consecutive closed steps with exactly one read-only call
  const stepsSinceBatchReminder = new WeakMap()
  const turnUserText = new WeakMap()
  const turnEditSeen = new WeakMap()
  const scopeReminderPending = new WeakMap()

  ctx.on('tools/result', (exec) => {
    const agent = exec?.agent
    if (agent === undefined || typeof exec.name !== 'string') return
    if (exec.name.startsWith(GRAPH_PREFIX)) stepUsedGraph.set(agent, true)
    else if (isCodeNavigation(exec.name, exec.arguments)) stepUsedNavigation.set(agent, true)
    const calls = stepCalls.get(agent) ?? { count: 0, readOnlyCount: 0 }
    calls.count += 1
    if (isReadOnlyLookup(exec.name, exec.arguments)) calls.readOnlyCount += 1
    stepCalls.set(agent, calls)
    if (isScopeCheckEnabled && EDIT_TOOLS.has(exec.name) && !turnEditSeen.get(agent)) {
      turnEditSeen.set(agent, true)
      if (!asksForChange(turnUserText.get(agent) ?? '')) scopeReminderPending.set(agent, true)
    }
  })

  ctx.on('agent/pre-step', async ({ agent, messages, step }, next) => {
    const decision = await next()
    if (decision.kind === 'reject') return decision
    // Close the previous step's accounting.
    if (stepUsedGraph.get(agent)) navigationStepsWithoutGraph.set(agent, 0)
    else if (stepUsedNavigation.get(agent)) navigationStepsWithoutGraph.set(agent, (navigationStepsWithoutGraph.get(agent) ?? 0) + 1)
    stepUsedGraph.delete(agent)
    stepUsedNavigation.delete(agent)
    const calls = stepCalls.get(agent)
    stepCalls.delete(agent)
    const isSingleLookup = calls?.count === 1 && calls.readOnlyCount === 1
    singleLookupStepRun.set(agent, isSingleLookup ? (singleLookupStepRun.get(agent) ?? 0) + 1 : 0)
    stepsSinceBatchReminder.set(agent, (stepsSinceBatchReminder.get(agent) ?? Infinity) + 1)

    const isNewRequest = hasUserRequest(messages)
    if (isNewRequest && step === 1) {
      turnUserText.set(agent, userTextOf(messages))
      turnEditSeen.delete(agent)
      scopeReminderPending.delete(agent)
    } else if (isNewRequest) {
      // A message typed mid-turn widens the request.
      turnUserText.set(agent, `${turnUserText.get(agent) ?? ''}\n${userTextOf(messages)}`)
    }

    const reminders = []
    if (step === 1 && isNewRequest) {
      reminders.push(TURN_REMINDER)
      navigationStepsWithoutGraph.set(agent, 0)
      say('turn reminder')
    } else if (driftSteps > 0 && (navigationStepsWithoutGraph.get(agent) ?? 0) >= driftSteps) {
      reminders.push(DRIFT_REMINDER)
      navigationStepsWithoutGraph.set(agent, 0)
      say(`drift reminder after ${driftSteps} navigation steps without a graph call`)
    }
    if (scopeReminderPending.get(agent)) {
      scopeReminderPending.delete(agent)
      reminders.push(SCOPE_REMINDER)
      say('scope reminder')
    }
    if (singleLookupSteps > 0 && (singleLookupStepRun.get(agent) ?? 0) >= singleLookupSteps &&
        stepsSinceBatchReminder.get(agent) >= batchReminderSpacing) {
      reminders.push(BATCH_REMINDER)
      singleLookupStepRun.set(agent, 0)
      stepsSinceBatchReminder.set(agent, 0)
      say(`batch reminder after ${singleLookupSteps} single-lookup steps`)
    }
    if (reminders.length === 0) return decision
    return { ...decision, messages: [...decision.messages, ...reminders.map(reminderMessage)] }
  })
}
