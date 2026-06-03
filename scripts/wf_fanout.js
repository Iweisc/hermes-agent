export const meta = {
  name: 'port-fanout-chunk',
  description: 'Port a chunk of dependency-bearing modules to Rust, written directly to disk',
  phases: [
    { title: 'Port', detail: 'one agent per module -> writes .rs to disk' },
    { title: 'Verify', detail: 'adversarial parity skeptic reads file' },
  ],
}

const REPO = '/Users/katcode/Desktop/Projects/hermes-agent'
const MODULES = MODULES_PLACEHOLDER

function destPath(m) {
  const crate = (m.crate === 'hermes') ? 'hermes' : 'hermes-core'
  return `crates/${crate}/src/${m.rs}`
}

const PORT_SCHEMA = {
  type: 'object',
  properties: {
    wrote_path: { type: 'string' },
    mod_name: { type: 'string' },
    public_api: { type: 'array', items: { type: 'string' } },
    line_count: { type: 'integer' },
    new_deps_required: { type: 'array', items: { type: 'string' } },
    cross_refs: { type: 'array', items: { type: 'string' }, description: 'other ported modules referenced via crate:: path' },
    behavior_notes: { type: 'string' },
  },
  required: ['wrote_path', 'mod_name', 'public_api', 'line_count'],
}
const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    divergences: { type: 'array', items: { type: 'object', properties: { severity: { type: 'string', enum: ['blocker','major','minor'] }, detail: { type: 'string' } }, required: ['severity','detail'] } },
    faithful: { type: 'boolean' },
  },
  required: ['divergences','faithful'],
}

phase('Port')
const ported = await pipeline(
  MODULES,
  (m) => {
    const dest = destPath(m)
    return agent(
      `Port the Python module ${REPO}/${m.py} (${m.loc} LOC) to native Rust.\n\n` +
      `WRITE the complete Rust source DIRECTLY to ${dest} via the Write tool (create it; it does not exist). Module name: \`${m.mod}\`.\n\n` +
      `Read the full Python source first; reproduce its behavior faithfully and idiomatically.\n` +
      `Hard rules:\n` +
      `- Write ONLY ${dest}. Do NOT edit lib.rs, main.rs, mod.rs, or any Cargo.toml — integration is done separately. Editing shared files corrupts the parallel run.\n` +
      `- No nested module directories.\n` +
      `- Use ONLY crates already available: serde, serde_json, serde_yaml, regex, chrono, base64, sha1/sha2, md5, hmac, aes, aes-gcm, flate2, tar, reqwest (blocking+json), url, dirs, libc, log, rusqlite, tokio, lettre, image, tungstenite. If you genuinely need another, still write the file and list it in new_deps_required.\n` +
      `- This module may depend on other hermes logic. Other ported modules live as flat files in crates/hermes-core/src (e.g. transports.rs, ag_anthropic_adapter.rs, mod_utils.rs) and crates/hermes/src. Reference them via \`crate::<modname>::\` where helpful and record those in cross_refs. If a dependency isn't ported yet, define a minimal local type/fn or take it as a parameter — do not block.\n` +
      `- Network modules: port request-construction + response-parsing with reqwest::blocking; keep API shapes exact.\n` +
      `- Include inline #[cfg(test)] unit tests for important cases. IMPORTANT: any std::env::set_var/remove_var call MUST be wrapped in an unsafe { } block (edition 2024).\n` +
      `- Make types/fns other modules need \`pub\`.\n\n` +
      `After writing, return metadata only — do NOT paste source into your response.`,
      { label: `port:${m.mod}`, phase: 'Port', schema: PORT_SCHEMA }
    ).then(r => ({ ...m, ...r, dest }))
  },
  (p, m) => agent(
    `Adversarially verify a Python->Rust port.\nOriginal: ${REPO}/${m.py}\nRust: ${REPO}/${p.wrote_path || p.dest}\n\n` +
    `Read BOTH. Flag BEHAVIORAL divergences (missing branches, wrong defaults, dropped errors, off-by-one, regex/ordering/overflow, missing serialized fields). Be skeptical; ignore pure style.`,
    { label: `verify:${m.mod}`, phase: 'Verify', schema: VERIFY_SCHEMA }
  ).then(v => ({ ...p, verdict: v }))
)

const results = ported.filter(Boolean)
return {
  count: results.length,
  written: results.map(r => ({
    py: r.py, wave: r.wave, mod: r.mod_name || r.mod, path: r.wrote_path || r.dest,
    crate: (r.crate === 'hermes') ? 'hermes' : 'hermes-core',
    lines: r.line_count, new_deps: r.new_deps_required || [],
    faithful: r.verdict?.faithful,
    blockers: (r.verdict?.divergences || []).filter(d => d.severity === 'blocker'),
    majors: (r.verdict?.divergences || []).filter(d => d.severity === 'major'),
  })),
}
