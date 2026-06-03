# Final cleanup phase — re-port / fix queue (accumulated correctness debt)

## PARTIAL mega-file ports (came back as skeletons, must be fully re-ported)
- crates/hermes-core/src/gw_run.rs        (gateway/run.py 14983 LOC -> 1864, 12%)
- crates/hermes-core/src/mod_run_agent.rs (run_agent.py 14461 -> 2710, 19%)
- crates/hermes/src/mod_cli.rs            (cli.py 12497 -> 2357, 19%)
- crates/hermes/src/cli_main.rs           (hermes_cli/main.py 10832 -> 1630, 15%)
- crates/hermes-core/src/cli_config.rs    (hermes_cli/config.py 4939 -> 1750, 35%)
- crates/hermes-core/src/gw_platforms_discord.rs (4697 -> 1827, 39%)
- crates/hermes-core/src/gw_platforms_telegram.rs (3661 -> 1685, 46%)

## Known real bugs (verifier blockers + failing tests) from earlier batches
- tool_path_security.rs: symlink traversal false-negative (SECURITY)
- mod_utils.rs: base_url_hostname bare-host parsing
- gw_wecom_crypto.rs: 4 aes tests failing
- gw_telegram_network.rs: is_private IP check diverges from Python ipaddress
- env_kimi_k2_parser.rs: extract_matches hard-breaks scan
- ag_model_metadata, cli_model_normalize, gw_helpers, agent_onboarding: failing unit tests

## Major divergences flagged (mega batch)
- mod_run_agent: serde_json key ordering (needs preserve_order feature or IndexMap)
- mod_cli: drops 10 of 15 default personalities; no env-var expansion in config
- gw_run: telegramize_command_mentions missing negative lookbehind
- gw_platforms_discord: document_ext_for vs os.path.splitext dotfile semantics
- gw_platforms_telegram: bold/strike/spoiler regex use DOTALL (Python doesn't)
- cli_main: wrong StepFun base URLs (dropped /step_plan path segment)

## Cross-cutting
- ~52 modules from the independent fan-out carried verifier divergences (see workflow outputs)
- NONE of the ported modules are wired to real callers yet (#[allow(dead_code)])
- serde_json preserve_order may be needed workspace-wide for faithful re-serialization

## Chunk 2 divergences (environments/tools)
- tool_environments_file_sync.rs: tar extraction lacks Python 3.12 'data' filter hardening (setuid/symlink) — SECURITY
- tool_environments_singularity.rs: String::truncate on byte index can panic on multibyte stderr; stderr not merged into stdout as Python does
- tool_environments_daytona.rs: missing initial forced sync + session init + _before_execute pre-step (FileSyncManager not wired)
- tool_environments_managed_modal.rs: float vs int JSON serialization (cpu/memoryMiB); empty-string error key fallthrough
- tool_website_policy.rs: url crate vs urllib divergence on odd hosts (spurious blocklist entries); serde_yaml YAML1.2 won't coerce yes/on/no/off booleans (config breakage)

## Chunk 3 divergences (tui_gateway / cli / cron)
- serde_json preserve_order needed: tui_event_publisher, mod_run_agent re-serialize with sorted keys (Python preserves insertion order). Enable serde_json "preserve_order" workspace-wide.
- tui_event_publisher.rs: compact JSON vs Python's ", "/": " separators (verbatim-rebroadcast contract)
- tui_entry.rs / tui_worker.rs: serde_json rejects NaN/Infinity (Python accepts); non-object frames crash Python but Rust replies-and-continues
- cli_model_switch.rs: model_sort_key tuple-vs-prefix ordering disagreement; HashMap alias reverse-lookup nondeterministic (Python dict insertion order)
- cli_commands.rs: splitn(2,' ') vs Python str.split(maxsplit=1) — breaks completion on tabs/double-spaces

## QUARANTINED modules (mod decl commented out — must fix to wire in)
### hermes-core (lib.rs):
  - ag_models_dev
  - gw_status
  - ag_codex_responses_adapter
  - ag_context_compressor
  - ag_copilot_acp_client
  - ag_image_routing
  - cli_config
  - cli_env_loader
  - cli_model_catalog
  - cli_providers
  - cli_timeouts
  - gw_config
  - gw_delivery
  - gw_platforms_base
  - gw_platforms_bluebubbles
  - gw_platforms_dingtalk
  - gw_platforms_email
  - gw_platforms_feishu
  - gw_platforms_homeassistant
  - gw_platforms_mattermost
  - gw_platforms_signal
  - gw_platforms_sms
  - gw_platforms_webhook
  - gw_platforms_wecom
  - gw_platforms_wecom_callback
  - gw_platforms_whatsapp
  - gw_restart
  - gw_run
  - gw_session
  - mod_batch_runner
  - mod_hermes_time
  - mod_mcp_serve
  - tool_hook_output_spill
  - tool_skills_sync
  - tui_entry
### hermes (main.rs):
  - tool_voice_mode
  - cli_callbacks
  - cli_cron
  - cli_debug
  - cli_gateway
  - cli_pairing
  - cli_skills_config
  - cli_skills_hub
  - cli_tips
  - cli_webhook
  - tool_approval
  - tool_browser_camofox
  - tool_browser_tool
  - tool_cronjob_tools
  - tool_environments_modal
  - tool_environments_vercel_sandbox
  - tool_image_generation_tool
  - tool_kanban_tools
  - tool_memory_tool
  - tool_process_registry
  - tool_session_search_tool
  - tool_skill_manager_tool
  - tool_skills_tool
  - tool_terminal_tool
  - tool_tirith_security
  - tool_tts_tool
  - tool_vision_tools
  - tool_yuanbao_tools
