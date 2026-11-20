# Naming conventions: self-documenting code

Variable, function, parameter, and property names must be self-documenting.
A reader should understand what a name holds or does without needing a comment
or having to trace where it came from.

Rules:
- Use full words, not abbreviations. Write `customerAccount` not `custAcct`,
  `response` not `res`, `error` not `err`, `index` not `idx`,
  `configuration` not `cfg`, `previousValue` not `prev`.
- Exceptions: industry-standard acronyms and identifiers that are more
  recognisable than their expansion (e.g. `cusip`, `isin`, `id`, `url`,
  `api`, `json`, `html`). Keep these.
- Single-letter names are only acceptable for trivial loop counters or
  conventional math/coordinate contexts (`x`, `y`). For loops over
  collections, name the item: `for (const valuation of valuations)`.
- Booleans should read as a question: `isExpired`, `hasCoupon`,
  `shouldRetry`, `canRedeem`.
- Functions should be verb phrases describing what they do:
  `calculateAccruedInterest`, `fetchLifecycleEvents`, not `calc` or `getData`.
- Include units or format where it matters: `timeoutMilliseconds`,
  `notionalAmountUsd`, `maturityDateIso`.
- Avoid vague names like `data`, `info`, `temp`, `result`, `value`, `item`,
  `obj`, `thing` unless the scope is tiny and the meaning is obvious.
- Prefer clarity over brevity. A longer name is fine if it removes ambiguity.

When editing existing code:
- Apply these rules to all new code you write.
- Only rename existing identifiers in code you're already changing, unless
  the user explicitly asks for a wider rename. Don't rename exported/public APIs
  without flagging it, since that can break consumers.
- "Least code" (ponytail) counts lines and abstractions, never name length: a
  longer, clearer name is never the thing to cut.
