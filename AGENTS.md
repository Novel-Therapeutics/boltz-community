# Agent instructions

## Scientific prediction reuse — human instruction, 12 October 2026

Before launching arms, inventory existing and in-flight prediction receipts and use a shared, durable cache keyed by the scientific request: exact sequence and target assembly/input hashes, prediction seed, model/settings, backend/precision and predictor source/runtime hashes. Arm name, population capacity and search RNG seed do not make an identical prediction a new scientific request.

Evaluate identical founders once and share the validated G0 baseline among compatible arms. Preserve each arm's independent population, ancestry, energy, selector and search RNG. Deduplicate repeated offspring in G1 and later generations too. Use a separate scoring identity so changed objectives/gates can reuse suitable raw predictions without silently reusing incompatible scores.

Verify applicable receipt, artifact, configuration, scorer/gate and reference hashes before reuse. Preserve completed, in-flight, failed and unknown attempts; never blindly repeat them. Let owned in-flight predictions finish and reconcile outcomes before reassigning work. Keep immutable historical results intact. Intentional prediction replication or a changed evidence version requires an explicit recorded purpose; do not disguise duplicate jobs as independent search replication.

For shared arm managers, persist full proposal/RNG manifests before scientific dispatch. A lost reply is not a missing prediction: keep CPU search state intact while reconciling native task, submission and scoring receipts. Make bookkeeping acknowledgements idempotent; never blindly repeat a prediction or completed raw CPU calculation.

## Branch naming — human instruction, 31 October 2026

Use descriptive branch names without a `codex` prefix or namespace. For example,
use `upstream-fast-path`, not `codex/upstream-fast-path`. Apply this to new
branches unless the user explicitly requests another name. Rename existing
branches only when requested; preserve their commits and uncommitted work.
