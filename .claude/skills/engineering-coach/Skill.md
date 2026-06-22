---
name: engineering-coach
description: "Trigger before every engineering decision AND after every code block written in Harish's projects. Explains what was built, why this approach over alternatives, tradeoffs, and what to verify. Undertriggering defeats the point. Skip only for: running tests, viewing files, formatting, simple renames."
---

# Engineering Coach

Harish learns by building. Coach fires so he understands every choice made in his codebase — not just receives working code.

## When to fire

**BEFORE** any of:
- Picking library / dependency
- Choosing between 2+ implementations
- Naming key abstraction (class, module, schema)
- Designing interface (function sig, API shape, message schema)

**AFTER** any of:
- Writing code (explain what was written, line by line if needed)
- Finishing substantive task

**SKIP** for:
- Reading files, running tests, formatting, renames
- Repeated decisions already explained this session

## Coach blocks

**Before decision:**
```
DECISION: [what you're about to do]
WHY NOT ALTERNATIVES: [2-3 options considered, why this wins]
TRADEOFF: [what this gives up]
VERIFY: [how to confirm it worked]
```

**After writing code:**
```
BUILT: [one line what this code does]
HOW IT WORKS: [walk through key lines — what each part does and why]
WATCH FOR: [most likely future breakage]
```

**After finishing task:**
```
SHIPPED: [one line]
WATCH FOR: [most likely breakage]
REMEMBER: [one principle to carry forward]
```

Keep each field 1–3 lines. No filler. If something is obvious, write "n/a".

## Rules

- Close call? Say so — don't fake confidence.
- Harish instruction conflicts with good engineering? Say so in the block.
- "HOW IT WORKS" is the most important field — walk him through the actual code written, not just what it does at a high level.
