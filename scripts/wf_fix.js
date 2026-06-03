export const meta = {
  name: 'fix-quarantined',
  description: 'Fix compile errors in quarantined hermes modules, one agent per module',
  phases: [{ title: 'Fix', detail: 'iterate cargo check per module until it compiles' }],
}

const REPO = '/Users/katcode/Desktop/Projects/hermes-agent'
const MODULES = MODULES_PLACEHOLDER

const FIX_SCHEMA = {
  type: 'object',
  properties: {
    mod_name: { type: 'string' },
    compiles: { type: 'boolean', description: 'true if `cargo build -p hermes-rs-cli` no longer reports errors originating in this file' },
    changes: { type: 'string', description: 'short summary of what was fixed' },
    remaining_blocker: { type: 'string', description: 'if not compiling, the root cause (e.g. depends on a still-broken module, needs a new dep, needs a design decision)' },
  },
  required: ['mod_name', 'compiles', 'changes'],
}

phase('Fix')
const results = await parallel(MODULES.map(m => () =>
  agent(
    `Fix the Rust compile errors in ${REPO}/${m.path} so the crate builds.\n\n` +
    `This is a machine-translated port of Python that doesn't yet compile. It is already declared (un-quarantined) in crates/hermes/src/main.rs behind #[allow(dead_code)].\n\n` +
    `Workflow:\n` +
    `1. Run: cargo build -p hermes-rs-cli 2>&1 | grep -A6 '${m.path}' to see THIS file's errors.\n` +
    `2. Read ${m.path} and fix the errors. Consult the original Python (find it under gateway/, tools/, hermes_cli/, agent/ by the module's base name) when behavior is unclear.\n` +
    `3. Re-run cargo build and iterate until no errors originate in ${m.path}.\n\n` +
    `Hard rules:\n` +
    `- Edit ONLY ${m.path}. Do NOT touch main.rs, lib.rs, Cargo.toml, or any other module file (other agents are fixing those in parallel — editing shared files corrupts the run).\n` +
    `- Only use crates already in crates/hermes/Cargo.toml.\n` +
    `- Preserve behavior; fix types/imports/borrows, don't delete functionality to silence errors. Wrap std::env::set_var/remove_var in unsafe{} (edition 2024).\n` +
    `- If this file genuinely cannot compile without changing ANOTHER file (e.g. it calls a function that doesn't exist yet, needs a new dep, or two duplicate types must be unified), STOP, set compiles=false, and describe it in remaining_blocker — do NOT edit other files.\n\n` +
    `The whole-crate build may still fail due to OTHER modules; that's fine. Success = no error points into ${m.path}.`,
    { label: `fix:${m.mod}`, phase: 'Fix', schema: FIX_SCHEMA }
  ).then(r => ({ ...m, ...r }))
))

const ok = results.filter(Boolean)
return {
  fixed: ok.filter(r => r.compiles).map(r => r.mod_name || r.mod),
  blocked: ok.filter(r => !r.compiles).map(r => ({ mod: r.mod_name || r.mod, blocker: r.remaining_blocker })),
}
