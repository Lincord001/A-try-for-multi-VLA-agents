from __future__ import annotations

import threading
from multiprocessing.connection import Client
from typing import Any

import numpy as np
import torch
from PIL import Image


class LocalPolicyBackend:
    """Thin adapter that exposes a uniform inference API for a local PI0 policy."""

    def __init__(self, *, policy, config: dict[str, Any], device: str, label: str):
        self.policy = policy
        self.config = config
        self.device = device
        self.label = label

    def reset(self) -> None:
        reset_fn = getattr(self.policy, "reset", None)
        if callable(reset_fn):
            reset_fn()

    def close(self) -> None:
        return None

    def infer_action_chunk(self, images_dict, state, task):
        image_size = int(self.config["image_size"])
        camera_keys = tuple(self.config["camera_keys"])

        resized_images = {
            key: Image.fromarray(np.asarray(images_dict[key])).resize(
                (image_size, image_size),
                resample=Image.BILINEAR,
            )
            for key in camera_keys
        }
        image_tensors = {
            key: torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1).to(torch.float32) / 255.0
            for key, img in resized_images.items()
        }
        image_tensors = {key: tensor.unsqueeze(0).to(self.device) for key, tensor in image_tensors.items()}
        state_tensor = torch.tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)

        batch = {
            "observation.state": state_tensor,
            "task": task.copy() if isinstance(task, list) else task,
        }
        for key, tensor in image_tensors.items():
            batch[f"observation.images.{key}"] = tensor

        with torch.no_grad():
            batch = self.policy.normalize_inputs(batch)
            images, img_masks = self.policy.prepare_images(batch)
            state_processed = self.policy.prepare_state(batch)
            lang_tokens, lang_masks = self.policy.prepare_language(batch)
            actions = self.policy.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state_processed
            )

            original_action_dim = self.policy.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]
            actions = self.policy.unnormalize_outputs({"action": actions})["action"]

            if self.policy.config.adapt_to_pi_aloha:
                actions = self.policy._pi_aloha_encode_actions(actions)

        chunk_np = actions[0].detach().cpu().numpy()
        return np.asarray(chunk_np, dtype=np.float32)


class RemotePolicyClient:
    """Client-side policy backend that talks to a persistent local policy server."""

    def __init__(self, *, mode: str, host: str, port: int, authkey: str):
        self.mode = str(mode)
        self.host = str(host)
        self.port = int(port)
        self.authkey = str(authkey).encode("utf-8")
        self._conn = None
        self._lock = threading.Lock()

    def __bool__(self) -> bool:
        return True

    def _ensure_connection(self):
        if self._conn is None:
            self._conn = Client((self.host, self.port), authkey=self.authkey)
        return self._conn

    def reset(self) -> None:
        return None

    def ping(self) -> None:
        with self._lock:
            conn = self._ensure_connection()
            conn.send({"type": "ping"})
            response = conn.recv()
        if not isinstance(response, dict) or response.get("status") != "ok":
            raise RuntimeError(f"policy server ping failed for mode={self.mode}: {response}")

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def infer_action_chunk(self, images_dict, state, task):
        payload = {
            "type": "infer",
            "mode": self.mode,
            "images": {key: np.asarray(value, dtype=np.uint8) for key, value in images_dict.items()},
            "state": np.asarray(state, dtype=np.float32),
            "task": task.copy() if isinstance(task, list) else task,
        }
        with self._lock:
            try:
                conn = self._ensure_connection()
                conn.send(payload)
                response = conn.recv()
            except (EOFError, BrokenPipeError, ConnectionResetError):
                self.close()
                conn = self._ensure_connection()
                conn.send(payload)
                response = conn.recv()

        if not isinstance(response, dict):
            raise RuntimeError(f"Invalid policy server response for mode={self.mode}: {response}")
        if response.get("status") != "ok":
            raise RuntimeError(str(response.get("error") or f"policy server error: {response}"))
        return np.asarray(response["actions"], dtype=np.float32)
