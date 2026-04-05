#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_ENDPOINT"] = "https://hf-mirror.com"

import torch

from deploy_v7_dual.config import (
    ARM_CONFIG,
    BASE_CONFIG,
    LOAD_ARM_MODEL,
    LOAD_BASE_MODEL,
    POLICY_SERVER_AUTHKEY,
    POLICY_SERVER_HOST,
    POLICY_SERVER_PORT,
)
from deploy_v7_dual.policy_loader import load_policy


def _parse_args():
    parser = argparse.ArgumentParser(description="Persistent local PI0 policy server")
    parser.add_argument("--host", default=POLICY_SERVER_HOST)
    parser.add_argument("--port", type=int, default=POLICY_SERVER_PORT)
    parser.add_argument("--authkey", default=POLICY_SERVER_AUTHKEY)
    return parser.parse_args()


def _serve_connection(conn, policies, infer_lock):
    try:
        while True:
            try:
                request = conn.recv()
            except EOFError:
                break

            if not isinstance(request, dict):
                conn.send({"status": "error", "error": "request must be a dict"})
                continue

            req_type = str(request.get("type", "")).strip().lower()
            if req_type == "ping":
                conn.send({"status": "ok"})
                continue

            if req_type != "infer":
                conn.send({"status": "error", "error": f"unsupported request type: {req_type}"})
                continue

            mode = str(request.get("mode", "")).strip().lower()
            backend = policies.get(mode)
            if backend is None:
                conn.send({"status": "error", "error": f"policy backend unavailable for mode={mode}"})
                continue

            try:
                with infer_lock:
                    actions = backend.infer_action_chunk(
                        request["images"],
                        request["state"],
                        request["task"],
                    )
                conn.send({"status": "ok", "actions": actions})
            except Exception as exc:
                conn.send({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main():
    args = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policies = {}

    if LOAD_ARM_MODEL:
        policies["arm"] = load_policy(ARM_CONFIG, device, label="ARM", emoji="🤖")
    if LOAD_BASE_MODEL:
        policies["base"] = load_policy(BASE_CONFIG, device, label="BASE", emoji="🚗")

    if not policies:
        raise RuntimeError("No policies were loaded. Check LOAD_ARM_MODEL / LOAD_BASE_MODEL.")

    infer_lock = threading.Lock()
    listener = Listener((args.host, args.port), authkey=args.authkey.encode("utf-8"))
    print(
        f"🧠 [POLICY SERVER] Listening on {args.host}:{args.port} "
        f"| backends={sorted(policies.keys())} | device={device}"
    )

    try:
        while True:
            conn = listener.accept()
            thread = threading.Thread(
                target=_serve_connection,
                args=(conn, policies, infer_lock),
                daemon=True,
            )
            thread.start()
    except KeyboardInterrupt:
        print("\n🛑 [POLICY SERVER] Interrupted by user.")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        listener.close()
        for backend in policies.values():
            try:
                backend.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
