// Unit test for plugins/dsh-plugin-fast-search/cbm-mcp-filter.mjs (the CBM result filter proxy): default limit,
// JSON-schema noise rows, project-name shortening and expansion, on synthetic CBM output (no binary needed).
// Run:   node scripts/test-cbm-mcp-filter.mjs        exits non-zero on any mismatch
import {
  encodedRootPrefix, shortenProjectNames, expandProjectName, rewriteCallArguments, filterResultText, filterCallResult,
} from '../plugins/dsh-plugin-fast-search/cbm-mcp-filter.mjs'

let failureCount = 0
const check = (label, actual, expected) => {
  const isMatch = JSON.stringify(actual) === JSON.stringify(expected)
  if (!isMatch) failureCount++
  console.log(`${isMatch ? 'ok  ' : 'FAIL'} ${label}: ${JSON.stringify(actual)}${isMatch ? '' : ` (want ${JSON.stringify(expected)})`}`)
}

const rootPrefix = encodedRootPrefix('/home/dev/Projects')
const options = { rootPrefix, defaultLimit: 10 }
check('encoded root prefix', rootPrefix, 'home-dev-Projects-')
check('empty root: no prefix', encodedRootPrefix(''), '')

check('limit added to search_graph', rewriteCallArguments('search_graph', { project: 'p', query: 'q' }, options).limit, 10)
check('limit added to trace_path', rewriteCallArguments('trace_path', { project: 'p', function_name: 'f' }, options).limit, 10)
check('explicit limit kept', rewriteCallArguments('search_code', { project: 'p', pattern: 'x', limit: 30 }, options).limit, 30)
check('no limit on get_code_snippet', rewriteCallArguments('get_code_snippet', { project: 'p', qualified_name: 'q' }, options).limit, undefined)
check('qualified_name expanded', rewriteCallArguments('get_code_snippet',
  { project: '@api-svc', qualified_name: '@api-svc.src.tenant.resolve' }, options),
  { project: 'home-dev-Projects-api-svc', qualified_name: 'home-dev-Projects-api-svc.src.tenant.resolve' })
check('function_name expanded', expandProjectName('@api-svc.src.a.b', rootPrefix), 'home-dev-Projects-api-svc.src.a.b')
check('compact ref "@0+x" untouched', expandProjectName('@0+src.a', rootPrefix), '@0+src.a')
check('plain name untouched', expandProjectName('resolveTenant', rootPrefix), 'resolveTenant')
check('round trip', expandProjectName(shortenProjectNames('home-dev-Projects-api-svc.src.a.f', rootPrefix), rootPrefix),
  'home-dev-Projects-api-svc.src.a.f')
check('file paths untouched', shortenProjectNames('/home/dev/Projects/api-svc/src/a.js', rootPrefix), '/home/dev/Projects/api-svc/src/a.js')

const searchGraphTree = [
  'results: 4  (cols: qn label file lines rank)',
  '  home-dev-Projects-api-svc.src.tenant.resolve Function src/tenant.js 3-27 -28.5',
  '  home-dev-Projects-api-svc.ops.pm2.app.PERF Variable ops/pm2/app.json 30-30 -17.2',
  '  home-dev-Projects-api-svc.ops.pm2.b.PERF Variable ops/pm2/b.json 31-31 -17.1',
  '  home-dev-Projects-api-svc.src.config.PERF Variable src/config.js 5-5 -17.0',
  'total: 112',
  'has_more: true',
].join('\n')
check('search_graph: noise rows dropped, names shortened', filterResultText('search_graph', searchGraphTree, rootPrefix), [
  'results: 2 (2 JSON-schema Variable rows hidden)  (cols: qn label file lines rank)',
  '  @api-svc.src.tenant.resolve Function src/tenant.js 3-27 -28.5',
  '  @api-svc.src.config.PERF Variable src/config.js 5-5 -17.0',
  'total: 112',
  'has_more: true',
].join('\n'))
const searchCodeCompact = [
  'results_refs: 2  (cols: id prefix)',
  '  0 home-dev-Projects-api-svc.',
  '  1 ops/pm2/',
  'results: 2  (cols: qn label file lines matches matches_omitted in out)',
  '  @0+src.a.f Function src/a.js 1-9 3 0 0 1',
  '  @0+ops.pm2.x.PERF Variable @1+x.json 30-30 "30" 0 0 0',
].join('\n')
check('search_code compact: noise row with a ref path dropped', filterResultText('search_code', searchCodeCompact, rootPrefix), [
  'results_refs: 2  (cols: id prefix)',
  '  0 @api-svc.',
  '  1 ops/pm2/',
  'results: 1 (1 JSON-schema Variable rows hidden)  (cols: qn label file lines matches matches_omitted in out)',
  '  @0+src.a.f Function src/a.js 1-9 3 0 0 1',
].join('\n'))
check('search_graph json format: noise dropped', JSON.parse(filterResultText('search_graph', JSON.stringify({ results: [
  { qn: 'home-dev-Projects-api-svc.a', label: 'Function', file: 'src/a.js' },
  { qn: 'home-dev-Projects-api-svc.b', label: 'Variable', file: 'ops/b.json' }] }), rootPrefix)),
  { results: [{ qn: '@api-svc.a', label: 'Function', file: 'src/a.js' }] })
check('get_code_snippet: no row filtering, names shortened',
  filterResultText('get_code_snippet', 'qualified_name: home-dev-Projects-api-svc.src.a.f\nlabel: Variable\n', rootPrefix),
  'qualified_name: @api-svc.src.a.f\nlabel: Variable\n')
check('call result: only text parts change', filterCallResult('trace_path', {
  content: [{ type: 'text', text: 'callers: 1  (cols: qn hop)\n  home-dev-Projects-api-svc.src.app 2' }, { type: 'image', data: 'x' }],
  isError: false }, rootPrefix),
  { content: [{ type: 'text', text: 'callers: 1  (cols: qn hop)\n  @api-svc.src.app 2' }, { type: 'image', data: 'x' }], isError: false })

console.log(failureCount === 0 ? 'all checks passed' : `${failureCount} check(s) failed`)
process.exit(failureCount === 0 ? 0 : 1)
