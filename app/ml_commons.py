"""Generic OpenSearch ML Commons helpers, shared by every AI-search POC.

These are model-agnostic: registering/deploying a pretrained model, polling the
deploy task, and finding an already-deployed model by name. Both the neural-sparse
bootstrap (`app.sparse.bootstrap`) and the dense bootstrap (`app.semantic.bootstrap`)
build on them, so the reuse-detection logic lives in exactly one place.
"""

from __future__ import annotations

import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"


def apply_ml_settings(client) -> None:
    """Loosen ML Commons guards so models run on a single local node."""
    print(
        "\n[ml] Applying ML Commons cluster settings "
        "(run on single node + raise memory threshold guard)..."
    )
    client.cluster.put_settings(
        body={
            "persistent": {
                "plugins.ml_commons.only_run_on_ml_node": False,
                "plugins.ml_commons.model_access_control_enabled": False,
                "plugins.ml_commons.native_memory_threshold": 99,
            }
        }
    )
    print("      done.")


def poll_task(client, task_id: str, *, attempts: int = 120, delay: int = 5) -> str:
    """Poll an ML task until COMPLETED; return its model_id."""
    for i in range(attempts):
        task = client.transport.perform_request("GET", f"/_plugins/_ml/tasks/{task_id}")
        state = task.get("state")
        print(f"      [{i * delay:>3}s] task state: {state}")
        if state == "COMPLETED":
            return task["model_id"]
        if state in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"ML task {task_id} {state}: {task.get('error', 'unknown error')}")
        time.sleep(delay)
    raise TimeoutError(f"ML task {task_id} did not complete in time")


def find_deployed_model(client, model_name: str) -> str | None:
    """Return the id of an already-deployed model with this name, if any.

    `.plugins-ml-model` stores each model as one metadata document PLUS N content
    chunks (chunk docs carry `chunk_number`/`total_chunks` but no `model_state`). We
    must exclude the chunks, or every match looks stateless and we'd re-register the
    model on every run. Only the metadata doc has `model_state`; its `_id` is the
    model_id.
    """
    try:
        res = client.transport.perform_request(
            "POST",
            "/_plugins/_ml/models/_search",
            body={
                "query": {
                    "bool": {
                        "must": [{"term": {"name.keyword": model_name}}],
                        "must_not": [{"exists": {"field": "chunk_number"}}],
                    }
                },
                "size": 20,
            },
        )
    except Exception:
        return None
    for hit in res.get("hits", {}).get("hits", []):
        src = hit.get("_source", {})
        if src.get("model_state") == "DEPLOYED":
            return hit["_id"]
    return None


def register_and_deploy(
    client, *, name: str, version: str, model_format: str
) -> tuple[str, bool]:
    """Register + deploy a pretrained model, reusing it if already deployed.

    Returns (model_id, reused).
    """
    print("\n[ml] Registering + deploying model...")
    print(f"      model : {name}")
    existing = find_deployed_model(client, name)
    if existing:
        print(f"      reusing already-deployed model: {existing}")
        wait_until_ready(client, existing)
        return existing, True

    resp = client.transport.perform_request(
        "POST",
        "/_plugins/_ml/models/_register?deploy=true",
        body={"name": name, "version": version, "model_format": model_format},
    )
    task_id = resp["task_id"]
    print(f"      task id: {task_id}")
    model_id = poll_task(client, task_id)
    print(f"      model deployed: {model_id}")
    wait_until_ready(client, model_id)
    return model_id, False


def _predict_ok(client, model_id: str) -> bool:
    """True if the model answers a tiny predict call (i.e. it is loaded for inference)."""
    try:
        client.transport.perform_request(
            "POST",
            f"/_plugins/_ml/models/{model_id}/_predict",
            body={"text_docs": ["ready?"]},
        )
        return True
    except Exception:
        return False


def wait_until_ready(client, model_id: str, *, attempts: int = 40, delay: int = 3) -> None:
    """Block until the model actually serves inference, not just reports DEPLOYED.

    A `register?deploy=true` task can report COMPLETED (and `model_state=DEPLOYED`)
    before the node has loaded the model, so the first ingest/query fails with
    "Model not ready yet." We probe with a tiny predict; if it fails we nudge an
    explicit `_deploy` (which reliably loads it) and poll until predict succeeds.
    """
    print(f"\n[ml] Ensuring model {model_id} is loaded for inference...")
    if _predict_ok(client, model_id):
        print("      ready.")
        return

    # DEPLOYED but not loaded — trigger an explicit deploy and wait for its task.
    print("      not loaded yet -> issuing explicit _deploy...")
    try:
        resp = client.transport.perform_request(
            "POST", f"/_plugins/_ml/models/{model_id}/_deploy"
        )
        deploy_task = resp.get("task_id")
        if deploy_task:
            poll_task(client, deploy_task)
    except Exception as exc:  # keep polling predict even if the nudge errors
        print(f"      _deploy nudge failed: {str(exc)[:80]}")

    for i in range(attempts):
        if _predict_ok(client, model_id):
            print("      ready.")
            return
        print(f"      [{i * delay:>3}s] still loading...")
        time.sleep(delay)
    raise TimeoutError(f"Model {model_id} did not become inference-ready in time")


def _set_env_var(lines: list[str], key: str, value: str) -> list[str]:
    """Replace `key=...` in .env lines, or append it if not present."""
    prefix = f"{key}="
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            lines[i] = f"{key}={value}"
            return lines
    lines.append(f"{key}={value}")
    return lines


def persist_env_var(key: str, value: str, env_file: Path = ENV_FILE) -> None:
    """Write `key=value` into the .env file (create it from .env.example if missing)."""
    print(f"\n[ml] Persisting {key} to {env_file.name}...")
    if not env_file.exists():
        example = env_file.with_name(".env.example")
        env_file.write_text(example.read_text() if example.exists() else "")
    lines = env_file.read_text().splitlines()
    lines = _set_env_var(lines, key, value)
    env_file.write_text("\n".join(lines) + "\n")
    print("      done.")
