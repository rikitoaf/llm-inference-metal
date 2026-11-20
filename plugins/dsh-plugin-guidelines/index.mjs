// dsh-plugin-guidelines: coding-behaviour rules for every agent, plus session handoff.
// Each enabled rule file (rules/*.md) becomes one static system-prompt section, placed after the first-party
// guidance and before the deployment persona suffix. The text never changes inside a session, so it sits in the
// cached prompt prefix: it costs one cold prefill (~2K tokens for all four files), not a prefill per step.
// Rule sources: rules/karpathy.md (forrestchang/andrej-karpathy-skills, MIT), rules/ponytail.md (DietrichGebert/
// ponytail AGENTS.md, MIT), rules/naming.md (the user's naming conventions), rules/handoff.md (handoff format).
// Handoff: /handoff [quick] [notes] asks the agent to write HANDOFF.md; /resume asks it to continue from one. When a
// session starts in a folder that already has HANDOFF.md, a one-paragraph notice points the agent at it. That notice
// is decided once per session and then frozen, so writing a new HANDOFF.md mid-session does not break the cache.
import { existsSync, readFileSync, statSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import Schema from '@deepseek-ai/schemastery'
import { createUserMessage } from '@deepseek-ai/dsh-llm'

export const name = 'guidelines'
export const inject = ['systemPrompt']

const pluginDirectory = dirname(fileURLToPath(import.meta.url))
const defaultRuleNames = ['karpathy', 'ponytail', 'naming', 'handoff']
// Between WEB_SURFACE (10100) and DEPLOYMENT_PERSONA_SUFFIX (10200) in dsh-system-prompt.
const defaultSectionOrder = 10150

export const Config = Schema.object({
  rules: Schema.array(Schema.string()).default(defaultRuleNames),
  ruleDirectory: Schema.string().default(''),
  sectionOrder: Schema.number().default(defaultSectionOrder),
  handoffFileName: Schema.string().default('HANDOFF.md'),
  isResumeNoticeEnabled: Schema.boolean().default(true),
  areCommandsEnabled: Schema.boolean().default(true),
  verbose: Schema.boolean().default(true),
})

export function loadRuleTexts(ruleNames, ruleDirectory, handoffFileName) {
  return ruleNames.map((ruleName) => {
    const rulePath = join(ruleDirectory, `${ruleName}.md`)
    const ruleText = readFileSync(rulePath, 'utf8').replaceAll('{{handoffFile}}', handoffFileName).trim()
    return { ruleName, ruleText }
  })
}

export function describeExistingHandoff(workingDirectory, handoffFileName) {
  if (!workingDirectory) return ''
  const handoffPath = join(workingDirectory, handoffFileName)
  if (!existsSync(handoffPath)) return ''
  const modifiedDateIso = statSync(handoffPath).mtime.toISOString().slice(0, 10)
  return `A handoff from an earlier session exists at ${handoffPath} (last written ${modifiedDateIso}). ` +
    'If the user asks to resume or continue, or the request clearly continues that work, read it first and follow ' +
    'its resume instructions. Otherwise leave it alone.'
}

export function buildHandoffRequest(rawInput, handoffFileName) {
  const commandArguments = rawInput.trim()
  const isQuick = /^quick\b/.test(commandArguments)
  const userNotes = (isQuick ? commandArguments.slice('quick'.length) : commandArguments).trim()
  const handoffKind = isQuick ? 'a quick handoff (Goal, Not yet done, Current state, Resume instructions only)' : 'a full handoff'
  return `Write ${handoffKind} to ${handoffFileName} in the working directory, in the session handoff format. ` +
    'Check git status first so Current state is accurate.' + (userNotes ? `\nNotes from the user: ${userNotes}` : '')
}

export function buildResumeRequest(handoffFileName) {
  return `Resume from ${handoffFileName} in the working directory: read it, check git status and git log -3 for ` +
    'drift since it was written, say in two lines what you will continue with, then continue.'
}

export function apply(ctx, configuration) {
  const logMessage = (message) => { if (configuration.verbose !== false) process.stderr.write(`[guidelines] ${message}\n`) }
  const handoffFileName = configuration.handoffFileName || 'HANDOFF.md'
  const ruleDirectory = configuration.ruleDirectory || join(pluginDirectory, 'rules')
  const sectionOrder = configuration.sectionOrder ?? defaultSectionOrder
  const ruleTexts = loadRuleTexts(configuration.rules ?? defaultRuleNames, ruleDirectory, handoffFileName)

  ruleTexts.forEach(({ ruleName, ruleText }, ruleIndex) => {
    ctx.effect(() => ctx.systemPrompt.section({
      name: `guidelines:${ruleName}`,
      order: sectionOrder + ruleIndex * 0.01,
      text: ruleText,
    }), `guidelines.section(${ruleName})`)
  })
  const totalCharacters = ruleTexts.reduce((sum, { ruleText }) => sum + ruleText.length, 0)
  logMessage(`rules ${ruleTexts.map(({ ruleName }) => ruleName).join(', ')} (${totalCharacters} chars)`)

  if (configuration.isResumeNoticeEnabled !== false) {
    const noticeBySession = new WeakMap()
    ctx.effect(() => ctx.systemPrompt.section({
      name: 'guidelines:handoff-notice',
      order: sectionOrder + 0.5,
      text: (promptContext) => {
        const session = promptContext.agent?.session
        if (session === undefined) return ''
        if (!noticeBySession.has(session)) {
          const noticeText = describeExistingHandoff(session.header?.cwd, handoffFileName)
          if (noticeText) logMessage(`found ${handoffFileName} in ${session.header.cwd}`)
          noticeBySession.set(session, noticeText)
        }
        return noticeBySession.get(session)
      },
    }), 'guidelines.section(handoff-notice)')
  }

  if (configuration.areCommandsEnabled === false) return
  const steerAgent = (agent, requestText) => agent.steer(createUserMessage({
    content: [{ type: 'text', text: requestText }],
    source: { kind: 'user' },
  }))
  ctx.inject(['commands'], (commandContext) => {
    commandContext.commands.register({
      name: 'handoff',
      description: `Write ${handoffFileName} so a new session can continue this work`,
      input: { hint: '[quick] [notes]' },
      handler: ({ agent, rawInput }) => {
        steerAgent(agent, buildHandoffRequest(rawInput, handoffFileName))
        return { kind: 'success', text: `Writing ${handoffFileName}.` }
      },
    })
    commandContext.commands.register({
      name: 'resume',
      description: `Continue from ${handoffFileName}`,
      handler: ({ agent, rawInput }) => {
        if (rawInput.trim()) return { kind: 'error', text: 'Usage: /resume (no arguments)' }
        steerAgent(agent, buildResumeRequest(handoffFileName))
        return { kind: 'success', text: `Resuming from ${handoffFileName}.` }
      },
    })
  })
}
