// Unit test for plugins/dsh-plugin-guidelines: drives apply() with a fake systemPrompt and command registry and
// checks the sections, the frozen handoff notice and the /handoff and /resume commands. Run:
//   node scripts/test-guidelines.mjs        exits non-zero on any mismatch
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { apply } from '../plugins/dsh-plugin-guidelines/index.mjs'

function createHarness(configuration) {
  const sections = []
  const commands = new Map()
  const ctx = {
    effect: (register) => register(),
    systemPrompt: { section: (section) => { sections.push(section); return () => {} } },
    inject: (_names, callback) => callback({ commands: { register: (command) => commands.set(command.name, command) } }),
  }
  apply(ctx, { verbose: false, ...configuration })
  return { sections, commands }
}

let failureCount = 0
const check = (label, actual, expected) => {
  const isMatch = JSON.stringify(actual) === JSON.stringify(expected)
  if (!isMatch) failureCount++
  console.log(`${isMatch ? 'ok  ' : 'FAIL'} ${label}: ${JSON.stringify(actual)}${isMatch ? '' : ` (want ${JSON.stringify(expected)})`}`)
}

const workingDirectory = mkdtempSync(join(tmpdir(), 'guidelines-test-'))
try {
  {
    const { sections, commands } = createHarness({})
    check('section names', sections.map((section) => section.name),
      ['guidelines:karpathy', 'guidelines:ponytail', 'guidelines:naming', 'guidelines:handoff', 'guidelines:handoff-notice'])
    check('orders sit before the persona suffix (10200)', sections.every((section) => section.order > 10100 && section.order < 10200), true)
    check('handoff placeholder filled', sections[3].text.includes('`HANDOFF.md`') && !sections[3].text.includes('{{'), true)
    check('karpathy comes before ponytail', sections[0].order < sections[1].order, true)
    check('ponytail defers to karpathy', sections[1].text.includes('follow Karpathy'), true)
    check('ponytail conflicts removed', ['Deletion beats addition', 'A vague request'].filter((line) => sections[1].text.includes(line)), [])
    check('rule texts are static strings', sections.slice(0, 4).every((section) => typeof section.text === 'string'), true)

    const notice = sections[4].text
    const session = { header: { cwd: workingDirectory } }
    check('no notice without an agent', notice({}), '')
    check('no notice without HANDOFF.md', notice({ agent: { session } }), '')
    writeFileSync(join(workingDirectory, 'HANDOFF.md'), '# Handoff: test\n')
    check('notice stays frozen for the same session', notice({ agent: { session } }), '')
    const newSession = { header: { cwd: workingDirectory } }
    check('new session sees HANDOFF.md', notice({ agent: { session: newSession } }).startsWith('A handoff from an earlier session exists at'), true)

    const steeredTexts = []
    const agent = { steer: (message) => steeredTexts.push(message.content[0].text) }
    check('commands registered', [...commands.keys()], ['handoff', 'resume'])
    check('/handoff succeeds', commands.get('handoff').handler({ agent, rawInput: '' }).kind, 'success')
    check('/handoff asks for a full handoff', steeredTexts[0].startsWith('Write a full handoff to HANDOFF.md'), true)
    commands.get('handoff').handler({ agent, rawInput: ' quick stuck on flaky test' })
    check('/handoff quick keeps notes', steeredTexts[1].includes('quick handoff') && steeredTexts[1].endsWith('Notes from the user: stuck on flaky test'), true)
    check('/resume with arguments errors', commands.get('resume').handler({ agent, rawInput: 'x' }).kind, 'error')
    commands.get('resume').handler({ agent, rawInput: '' })
    check('/resume steers', steeredTexts[2].startsWith('Resume from HANDOFF.md'), true)
  }
  {
    const { sections, commands } = createHarness({ rules: ['naming'], isResumeNoticeEnabled: false, areCommandsEnabled: false })
    check('subset of rules only', sections.map((section) => section.name), ['guidelines:naming'])
    check('commands off', commands.size, 0)
  }
} finally {
  rmSync(workingDirectory, { recursive: true, force: true })
}

console.log(failureCount ? `\n${failureCount} failure(s)` : '\nall passed')
process.exit(failureCount ? 1 : 0)
