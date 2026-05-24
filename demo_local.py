#!/usr/bin/env python3
"""Local desktop launcher for the DecodeShare interactive steering demo."""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Tkinter is required for demo_local.py. Install python3-tk or use demo/app.py.") from exc


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.app import (  # noqa: E402
    CHAT_SESSIONS,
    DEFAULT_CHAT_CACHE,
    DEFAULT_CHAT_MODEL,
    EXAMPLE_CONFIGS,
    VECTOR_MODES,
    VECTOR_PRESETS,
    chat_once,
    default_device_choice,
    default_dtype_choice,
    default_max_new_tokens,
    load_example_config,
    prepare_chat_state,
)


History = List[Dict[str, str]]


def parse_args() -> argparse.Namespace:
    device = default_device_choice()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--device", default=device, choices=["cuda", "cpu", "mps"])
    parser.add_argument("--dtype", default=default_dtype_choice(device), choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--basis_k", type=int, default=24)
    parser.add_argument("--calib_max_new_tokens", type=int, default=6)
    parser.add_argument("--steer_max_new_tokens", type=int, default=6)
    parser.add_argument("--max_prompt_tokens", type=int, default=384)
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cache", default=str(DEFAULT_CHAT_CACHE.relative_to(REPO_ROOT)))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--no_save_cache", action="store_true")
    parser.add_argument("--auto_init", action="store_true")
    return parser.parse_args()


class QueueProgress:
    def __init__(self, events: "queue.Queue[tuple]") -> None:
        self.events = events

    def __call__(self, value: float, desc: str = "") -> None:
        self.events.put(("progress", float(value), str(desc)))


class LocalDemoApp:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self.session_id = ""
        self.worker: Optional[threading.Thread] = None
        self.baseline_history: History = []
        self.prefill_history: History = []
        self.decode_history: History = []

        self.model_var = tk.StringVar(value=args.model)
        self.device_var = tk.StringVar(value=args.device)
        self.dtype_var = tk.StringVar(value=args.dtype)
        self.layer_var = tk.IntVar(value=args.layer)
        self.basis_k_var = tk.IntVar(value=args.basis_k)
        self.calib_tokens_var = tk.IntVar(value=args.calib_max_new_tokens)
        self.steer_tokens_var = tk.IntVar(value=args.steer_max_new_tokens)
        self.max_prompt_var = tk.IntVar(value=args.max_prompt_tokens)
        self.system_var = tk.StringVar(value=args.system)
        self.seed_var = tk.IntVar(value=args.seed)
        self.cache_var = tk.StringVar(value=args.cache)
        self.use_cache_var = tk.BooleanVar(value=not args.no_cache)
        self.save_cache_var = tk.BooleanVar(value=not args.no_save_cache)
        self.local_files_var = tk.BooleanVar(value=args.local_files_only)
        self.trust_remote_var = tk.BooleanVar(value=args.trust_remote_code)

        first_example = next(iter(EXAMPLE_CONFIGS))
        example = EXAMPLE_CONFIGS[first_example]
        self.example_var = tk.StringVar(value=first_example)
        self.preset_var = tk.StringVar(value=str(example["preset"]))
        self.mode_var = tk.StringVar(value=str(example["vector_mode"]))
        self.alpha_var = tk.DoubleVar(value=float(example["alpha"]))
        self.beta_var = tk.DoubleVar(value=float(example["beta"]))
        self.inject_first_n_var = tk.IntVar(value=int(example["inject_first_n"]))
        self.max_new_tokens_var = tk.IntVar(
            value=min(int(example["max_new_tokens"]), default_max_new_tokens(args.device))
        )
        self.status_var = tk.StringVar(value="Not initialized.")
        self.progress_var = tk.DoubleVar(value=0.0)

        self.root.title("DecodeShare Local Demo")
        self.root.geometry("1320x820")
        self.root.minsize(980, 660)
        self._build_ui()
        self._apply_example(first_example)
        self._set_busy(False)
        self.root.after(100, self._drain_events)
        if args.auto_init:
            self.root.after(200, self.start_initialize)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        config = ttk.LabelFrame(self.root, text="Runtime")
        config.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        for col in range(8):
            config.columnconfigure(col, weight=1)

        self._entry(config, "Model", self.model_var, 0, 0, width=36)
        self._combo(config, "Device", self.device_var, ["cuda", "cpu", "mps"], 0, 2, width=8)
        self._combo(config, "Dtype", self.dtype_var, ["fp16", "bf16", "fp32"], 0, 3, width=8)
        self._spin(config, "Layer", self.layer_var, 0, 4, 0, 96, width=6)
        self._spin(config, "Basis dim", self.basis_k_var, 0, 5, 1, 256, width=6)
        self._entry(config, "Cache", self.cache_var, 0, 6, width=32, columnspan=2)

        self._entry(config, "System", self.system_var, 1, 0, width=36)
        self._spin(config, "Basis tokens", self.calib_tokens_var, 1, 2, 1, 128, width=6)
        self._spin(config, "Vector tokens", self.steer_tokens_var, 1, 3, 1, 128, width=6)
        self._spin(config, "Max prompt", self.max_prompt_var, 1, 4, 64, 4096, width=7)
        self._spin(config, "Seed", self.seed_var, 1, 5, 0, 1_000_000, width=8)
        ttk.Checkbutton(config, text="Use cache", variable=self.use_cache_var).grid(
            row=2, column=6, sticky="w", padx=6, pady=4
        )
        ttk.Checkbutton(config, text="Save cache", variable=self.save_cache_var).grid(
            row=2, column=7, sticky="w", padx=6, pady=4
        )
        ttk.Checkbutton(config, text="Local files only", variable=self.local_files_var).grid(
            row=3, column=6, sticky="w", padx=6, pady=4
        )
        ttk.Checkbutton(config, text="Trust remote code", variable=self.trust_remote_var).grid(
            row=3, column=7, sticky="w", padx=6, pady=4
        )

        controls = ttk.LabelFrame(self.root, text="Chat")
        controls.grid(row=1, column=0, sticky="ew", padx=10, pady=6)
        controls.columnconfigure(0, weight=3)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)
        controls.columnconfigure(3, weight=1)
        controls.columnconfigure(4, weight=1)

        ttk.Label(controls, text="Prompt").grid(row=0, column=0, sticky="w", padx=6, pady=(6, 0))
        self.prompt_text = tk.Text(controls, height=4, wrap="word", undo=True)
        self.prompt_text.grid(row=1, column=0, rowspan=3, sticky="nsew", padx=6, pady=4)
        self.prompt_text.bind("<Control-Return>", lambda _event: self.start_send())

        example_combo = self._combo(
            controls, "Example", self.example_var, list(EXAMPLE_CONFIGS), 0, 1, width=28
        )
        example_combo.bind("<<ComboboxSelected>>", lambda _event: self.use_selected_example())
        self.use_example_button = ttk.Button(controls, text="Use Example", command=self.use_selected_example)
        self.use_example_button.grid(row=1, column=1, sticky="ew", padx=6, pady=4)
        self._combo(controls, "Preset", self.preset_var, ["None"] + list(VECTOR_PRESETS), 0, 2, width=18)
        self._combo(controls, "Vector mode", self.mode_var, VECTOR_MODES, 0, 3, width=22)
        self._spin(controls, "Max tokens", self.max_new_tokens_var, 0, 4, 8, 192, increment=4, width=8)
        self._spin(controls, "Alpha", self.alpha_var, 2, 1, -6.0, 6.0, increment=0.25, width=8)
        self._spin(controls, "Beta", self.beta_var, 2, 2, 0.0, 1.0, increment=0.05, width=8)
        self._spin(controls, "Inject steps", self.inject_first_n_var, 2, 3, 1, 256, width=8)

        self.init_button = ttk.Button(controls, text="Initialize", command=self.start_initialize)
        self.init_button.grid(row=3, column=1, sticky="ew", padx=6, pady=6)
        self.send_button = ttk.Button(controls, text="Send", command=self.start_send)
        self.send_button.grid(row=3, column=2, sticky="ew", padx=6, pady=6)
        self.clear_button = ttk.Button(controls, text="Clear", command=self.clear_chat)
        self.clear_button.grid(row=3, column=3, sticky="ew", padx=6, pady=6)
        self.progress = ttk.Progressbar(controls, variable=self.progress_var, maximum=1.0)
        self.progress.grid(row=3, column=4, sticky="ew", padx=6, pady=6)

        responses = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        responses.grid(row=2, column=0, sticky="nsew", padx=10, pady=6)
        self.baseline_text = self._response_pane(responses, "Baseline")
        self.prefill_text = self._response_pane(responses, "Prefill-estimated vector")
        self.decode_text = self._response_pane(responses, "Decode-estimated vector")

        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        status.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))

    def _entry(
        self,
        parent: tk.Widget,
        label: str,
        variable: tk.Variable,
        row: int,
        column: int,
        *,
        width: int,
        columnspan: int = 1,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row * 2, column=column, sticky="w", padx=6, pady=(6, 0))
        ttk.Entry(parent, textvariable=variable, width=width).grid(
            row=row * 2 + 1, column=column, columnspan=columnspan, sticky="ew", padx=6, pady=4
        )

    def _combo(
        self,
        parent: tk.Widget,
        label: str,
        variable: tk.StringVar,
        values: List[str],
        row: int,
        column: int,
        *,
        width: int,
    ) -> ttk.Combobox:
        ttk.Label(parent, text=label).grid(row=row * 2, column=column, sticky="w", padx=6, pady=(6, 0))
        combo = ttk.Combobox(parent, textvariable=variable, values=values, width=width, state="readonly")
        combo.grid(row=row * 2 + 1, column=column, sticky="ew", padx=6, pady=4)
        return combo

    def _spin(
        self,
        parent: tk.Widget,
        label: str,
        variable: tk.Variable,
        row: int,
        column: int,
        from_: float,
        to: float,
        *,
        increment: float = 1,
        width: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row * 2, column=column, sticky="w", padx=6, pady=(6, 0))
        ttk.Spinbox(
            parent,
            textvariable=variable,
            from_=from_,
            to=to,
            increment=increment,
            width=width,
        ).grid(row=row * 2 + 1, column=column, sticky="ew", padx=6, pady=4)

    def _response_pane(self, parent: ttk.PanedWindow, label: str) -> tk.Text:
        frame = ttk.LabelFrame(parent, text=label)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        text = tk.Text(frame, wrap="word", state="disabled", padx=8, pady=8)
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        parent.add(frame, weight=1)
        return text

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.init_button.configure(state=state)
        self.send_button.configure(state=state)
        self.clear_button.configure(state=state)
        self.use_example_button.configure(state=state)

    def _start_worker(self, target: Any) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        self._set_busy(True)
        self.worker = threading.Thread(target=target, daemon=True)
        self.worker.start()

    def start_initialize(self) -> None:
        config = {
            "model": self.model_var.get().strip(),
            "device": self.device_var.get().strip(),
            "dtype": self.dtype_var.get().strip(),
            "layer": int(self.layer_var.get()),
            "basis_k": int(self.basis_k_var.get()),
            "calib_max_new_tokens": int(self.calib_tokens_var.get()),
            "steer_max_new_tokens": int(self.steer_tokens_var.get()),
            "max_prompt_tokens": int(self.max_prompt_var.get()),
            "system": self.system_var.get(),
            "seed": int(self.seed_var.get()),
            "local_files_only": bool(self.local_files_var.get()),
            "trust_remote_code": bool(self.trust_remote_var.get()),
            "cache_path": self.cache_var.get().strip(),
            "use_cache": bool(self.use_cache_var.get()),
            "save_cache": bool(self.save_cache_var.get()),
        }

        def work() -> None:
            try:
                self.events.put(("progress", 0.0, "Loading model and DecodeShare vectors"))
                state, status = prepare_chat_state(
                    model=config["model"],
                    device=config["device"],
                    dtype=config["dtype"],
                    layer=config["layer"],
                    basis_k=config["basis_k"],
                    calib_max_new_tokens=config["calib_max_new_tokens"],
                    steer_max_new_tokens=config["steer_max_new_tokens"],
                    max_prompt_tokens=config["max_prompt_tokens"],
                    system=config["system"],
                    seed=config["seed"],
                    local_files_only=config["local_files_only"],
                    trust_remote_code=config["trust_remote_code"],
                    cache_path=config["cache_path"],
                    use_cache=config["use_cache"],
                    save_cache=config["save_cache"],
                    progress=QueueProgress(self.events),
                )
                session_id = uuid4().hex
                CHAT_SESSIONS[session_id] = state
                self.events.put(("initialized", session_id, status, state["basis_info"], state["config"]))
            except Exception as exc:
                self.events.put(("error", f"Initialization failed: {exc}", traceback.format_exc()))

        self.start_status("Initializing...")
        self._start_worker(work)

    def start_send(self) -> None:
        message = self.prompt_text.get("1.0", "end").strip()
        if not message:
            self.start_status("Enter a prompt first.")
            return
        if not self.session_id:
            self.start_status("Initialize the demo first.")
            return
        chat_config = {
            "preset": self.preset_var.get(),
            "mode": self.mode_var.get(),
            "alpha": float(self.alpha_var.get()),
            "beta": float(self.beta_var.get()),
            "inject_first_n": int(self.inject_first_n_var.get()),
            "max_new_tokens": int(self.max_new_tokens_var.get()),
        }
        baseline_history = list(self.baseline_history)
        prefill_history = list(self.prefill_history)
        decode_history = list(self.decode_history)

        def work() -> None:
            try:
                result = chat_once(
                    self.session_id,
                    message,
                    baseline_history,
                    prefill_history,
                    decode_history,
                    chat_config["preset"],
                    chat_config["mode"],
                    chat_config["alpha"],
                    chat_config["beta"],
                    chat_config["inject_first_n"],
                    chat_config["max_new_tokens"],
                    progress=QueueProgress(self.events),
                )
                self.events.put(("chat_result", result))
            except Exception as exc:
                self.events.put(("error", f"Generation failed: {exc}", traceback.format_exc()))

        self.start_status("Generating...")
        self._start_worker(work)

    def clear_chat(self) -> None:
        self.baseline_history = []
        self.prefill_history = []
        self.decode_history = []
        self._set_response(self.baseline_text, "")
        self._set_response(self.prefill_text, "")
        self._set_response(self.decode_text, "")
        self.start_status("Cleared chat history.")

    def use_selected_example(self) -> None:
        self._apply_example(self.example_var.get())

    def _apply_example(self, label: str) -> None:
        prompt, preset, alpha, max_tokens, mode, beta, inject_first_n, status = load_example_config(
            label
        )
        self.baseline_history = []
        self.prefill_history = []
        self.decode_history = []
        self._set_response(self.baseline_text, "")
        self._set_response(self.prefill_text, "")
        self._set_response(self.decode_text, "")
        self.prompt_text.delete("1.0", "end")
        self.prompt_text.insert("1.0", prompt)
        self.preset_var.set(preset)
        self.alpha_var.set(alpha)
        self.max_new_tokens_var.set(max_tokens)
        self.mode_var.set(mode)
        self.beta_var.set(beta)
        self.inject_first_n_var.set(inject_first_n)
        self.start_status(f"{status}. Cleared chat history.")

    def start_status(self, text: str) -> None:
        self.status_var.set(text)

    def _drain_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event[0]
            if kind == "progress":
                _, value, desc = event
                self.progress_var.set(max(0.0, min(1.0, float(value))))
                if desc:
                    self.status_var.set(str(desc))
            elif kind == "initialized":
                _, session_id, status, basis_info, config = event
                self.session_id = session_id
                self.progress_var.set(1.0)
                self.status_var.set(self._ready_status(status, basis_info, config))
                self._set_busy(False)
            elif kind == "chat_result":
                _, result = event
                (
                    self.baseline_history,
                    self.prefill_history,
                    self.decode_history,
                    _message,
                    status,
                ) = result
                self.prompt_text.delete("1.0", "end")
                self._set_response(self.baseline_text, self._render_history(self.baseline_history))
                self._set_response(self.prefill_text, self._render_history(self.prefill_history))
                self._set_response(self.decode_text, self._render_history(self.decode_history))
                self.progress_var.set(1.0)
                self.status_var.set(status)
                self._set_busy(False)
            elif kind == "error":
                _, summary, detail = event
                self.progress_var.set(0.0)
                self.status_var.set(summary)
                self._set_busy(False)
                messagebox.showerror("DecodeShare Local Demo", f"{summary}\n\n{detail}")
        self.root.after(100, self._drain_events)

    def _ready_status(self, status: str, basis_info: Dict[str, Any], config: Dict[str, Any]) -> str:
        model = config.get("model", "")
        layer = config.get("layer", "")
        basis_dim = basis_info.get("basis_dim", "")
        n_states = basis_info.get("n_states", "")
        prefix = f"Ready: {model} | layer={layer} | basis={basis_dim} | states={n_states}"
        return f"{prefix} | {status}" if status else prefix

    def _set_response(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")
        widget.see("end")

    def _render_history(self, history: History) -> str:
        chunks = []
        for item in history:
            role = item.get("role", "assistant").capitalize()
            content = item.get("content", "")
            chunks.append(f"{role}:\n{content}")
        return "\n\n".join(chunks)


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    LocalDemoApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
