// Unit test for plugins/dsh-plugin-fast-search/graph-first-reminder.mjs: drives apply() with a fake ctx (a handlers
// map for `tools/result` and `agent/pre-step`) through simulated turns and checks which reminders are injected.
// Run:   node scripts/test-graph-first-reminder.mjs        exits non-zero on any mismatch
import {
  apply, asksForChange, isReadOnlyLookup,
  TURN_REMINDER, DRIFT_REMINDER, SCOPE_REMINDER, BATCH_REMINDER,
} from '../plugins/dsh-plugin-fast-search/graph-first-reminder.mjs'

let failureCount = 0
const check = (label, actual, expected) => {
  const isMatch = JSON.stringify(actual) === JSON.stringify(expected)
  if (!isMatch) failureCount++
  console.log(`${isMatch ? 'ok  ' : 'FAIL'} ${label}: ${JSON.stringify(actual)}${isMatch ? '' : ` (want ${JSON.stringify(expected)})`}`)
}

const REMINDER_NAMES = new Map([[TURN_REMINDER, 'turn'], [DRIFT_REMINDER, 'drift'], [SCOPE_REMINDER, 'scope'], [BATCH_REMINDER, 'batch']])

// A simulated agent: typed(text) starts a turn; step(calls) runs one step's pre-step and then its tool results.
// Each pre-step returns the names of the reminders it injected.
function createSession(configuration = {}) {
  const handlers = new Map()
  const ctx = { on: (event, handler) => handlers.set(event, handler) }
  apply(ctx, { verbose: false, ...configuration })
  const agent = {}
  let stepNumber = 0
  let pendingUserText
  const preStep = async () => {
    stepNumber += 1
    const messages = pendingUserText === undefined ? []
      : [{ role: 'user', source: { kind: 'user' }, content: [{ type: 'text', text: pendingUserText }] }]
    pendingUserText = undefined
    const decision = await handlers.get('agent/pre-step')({ agent, messages, step: stepNumber },
      async () => ({ kind: 'continue', messages: [] }))
    return decision.messages.map((message) => REMINDER_NAMES.get(message.content[0].text) ?? '?')
  }
  const toolResult = (name, args = {}) => handlers.get('tools/result')({ agent, name, arguments: args, callId: 'c' }, {})
  return {
    typed(text) { pendingUserText = text; stepNumber = 0 },
    async step(calls = []) {
      const injected = await preStep()
      for (const [name, args] of calls) check(`tools/result returns nothing (never blocks) for ${name}`, toolResult(name, args), undefined)
      return injected
    },
  }
}

const read = ['read', { file_path: '/repo/src/a.js' }]
const grep = ['grep', { pattern: 'x' }]
const graph = ['mcp__cbm__search_code', { project: 'p', pattern: 'x' }]
const edit = ['edit', { file_path: '/repo/src/a.js' }]
const bashNav = ['bash', { command: 'grep -rn foo src' }]
const bashRun = ['bash', { command: 'node inspect.js' }]

// --- turn reminder and rule 5
{
  check('turn reminder carries the scope rule', TURN_REMINDER.includes('5. Scope:'), true)
  const session = createSession({})
  session.typed('Explain how the export caps work.')
  check('step 1 of a typed request: turn reminder', await session.step([graph]), ['turn'])
  check('step 2: nothing', await session.step(), [])
}

// --- drift reminder (unchanged behaviour)
{
  const session = createSession({ driftSteps: 3, singleLookupSteps: 0 })
  session.typed('Find where the cache key is built.')
  await session.step([bashNav])
  await session.step([bashNav])
  await session.step([bashNav])
  check('drift reminder after 3 navigation steps', await session.step([bashNav]), ['drift'])
  const other = createSession({ driftSteps: 3, singleLookupSteps: 0 })
  other.typed('Run the inspector.')
  for (let index = 0; index < 4; index++) await other.step([bashRun])
  check('script runs do not count as navigation', await other.step(), [])
}

// --- scope check
{
  const session = createSession({ singleLookupSteps: 0 })
  session.typed('Why does the bulk upload never take the DZ branch?')
  await session.step([read, read])
  check('analysis request, first edit: scope reminder at the next step', await session.step([edit]), [])
  check('scope reminder injected once', await session.step([edit]), ['scope'])
  check('second edit in the same turn: no reminder', await session.step(), [])
  session.typed('Now fix it.')
  check('new turn with an action verb: turn reminder only', await session.step([edit]), ['turn'])
  check('edit after "fix": no scope reminder', await session.step(), [])
}
{
  const session = createSession({ singleLookupSteps: 0 })
  session.typed('did you change anything in postgres settings, which can cause having an issue on running local db?')
  await session.step([edit])
  check('question containing "change": scope reminder', await session.step(), ['scope'])
}
{
  const session = createSession({ isScopeCheckEnabled: false, singleLookupSteps: 0 })
  session.typed('What does this function return?')
  await session.step([edit])
  check('scope check disabled: no reminder', await session.step(), [])
}
check('asksForChange: imperative', asksForChange('Add a wkn field to spi_extra.'), true)
check('asksForChange: polite request', asksForChange('Can you add WKN to the dropdown?'), true)
check('asksForChange: plan approval', asksForChange('yes'), true)
check('asksForChange: "do it"', asksForChange('looks right, do it'), true)
check('asksForChange: question with a verb', asksForChange('the change you made should it be a generalized change right?'), false)
check('asksForChange: analysis', asksForChange('Explain how the tenant key is resolved'), false)
check('asksForChange: "address" is not "add"', asksForChange('Explain the address lookup'), false)
check('asksForChange: empty', asksForChange(''), false)

// --- batch reminder
check('read-only lookups', [read, grep, graph, bashNav, edit, bashRun].map(([name, args]) => isReadOnlyLookup(name, args)),
  [true, true, true, true, false, false])
{
  const session = createSession({})
  session.typed('Trace the request flow.')
  check('step 1: turn reminder', await session.step([read]), ['turn'])
  check('after one single-lookup step: nothing', await session.step([grep]), [])
  check('after two single-lookup steps: batch reminder', await session.step([graph]), ['batch'])
  check('spacing: not again at step 4', await session.step([read]), [])
  check('spacing: not again at step 5', await session.step([read]), [])
  check('spacing: not again at step 6', await session.step([read]), [])
  check('spacing: not again at step 7', await session.step([read]), [])
  check('spacing: not again at step 8', await session.step([read]), [])
  check('6 steps later: batch reminder again', await session.step([read]), ['batch'])
}
{
  const session = createSession({})
  session.typed('Trace the request flow, then fix the timeout.')
  await session.step([read])
  await session.step([read, grep])
  check('a batched step resets the run', await session.step([read]), [])
  check('one single-lookup step after the reset: nothing', await session.step([read]), [])
  check('second single-lookup step: batch reminder', await session.step([edit]), ['batch'])
  check('an edit step is not a lookup', await session.step([read]), [])
}

console.log(failureCount === 0 ? 'all checks passed' : `${failureCount} check(s) failed`)
process.exit(failureCount === 0 ? 0 : 1)
