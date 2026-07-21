# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for a security problem.

Report it privately through
[GitHub Security Advisories](https://github.com/harishchaurasia/compass-arch/security/advisories/new),
or email **harishchaurasia143@gmail.com**. Include what you found, how to reproduce
it, and what an attacker could do with it. Expect an initial reply within about a
week; this is a research project maintained by one person, so please be patient.

## Supported versions

Compass is pre-1.0 research code. Only `main` receives fixes. There are no
backported patches for older tags.

## Scope

In scope:

- Code execution or injection via task definitions, trace files, or MCP tool
  arguments.
- The MCP bridge (`compass/mcp/bridge.py`) mishandling untrusted server output.
- Anything that leaks credentials from the environment into logs, traces, or
  `results/trials.db`.

Out of scope:

- An agent taking a wrong or destructive action. That is the *subject* of this
  research, not a vulnerability. Compass reduces those failures on some models and
  does not eliminate them - see [FINDINGS.md](FINDINGS.md).
- Prompt injection that makes a model misbehave without crossing a trust boundary
  in this codebase.

## Running Compass safely

- Compass drives real MCP servers that can **write, move, and delete files**. Point
  the demos at a scratch directory, never a real working tree.
- Keep API keys in a gitignored `.env` (see `.env.example`). Never commit them.
- Use a throwaway, minimally-scoped token for the GitHub MCP demo. It exposes
  high-risk tools such as `merge_pull_request` and `push_files`.
- `results/trials.db` holds full model traces. It is gitignored by design - check
  before sharing it.
