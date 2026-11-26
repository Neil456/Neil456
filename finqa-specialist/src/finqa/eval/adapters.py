"""Model adapters: local Hugging Face (CPU/GPU, optional LoRA), OpenAI-compatible HTTP
(vLLM / llama.cpp server / OpenAI), Anthropic Messages API, Google Gemini API.

Every adapter returns, per request: text, input_tokens, output_tokens, latency_s,
finish_reason, and (API adapters) actual list-price cost. API adapters never read
endpoint/base-URL settings implicitly from generic environment variables (to avoid
accidentally routing through unrelated proxies); keys are read only from
FINQA_OPENAI_API_KEY / FINQA_ANTHROPIC_API_KEY / FINQA_GEMINI_API_KEY (falling back to
the provider's standard key variable) and are never logged.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


@dataclass
class GenConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    enable_thinking: bool = False
    reasoning_effort: str | None = None  # API reasoning controls, recorded verbatim
    batch_size: int = 4


@dataclass
class GenResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    finish_reason: str = ""
    cost_usd: float | None = None
    extra: dict = field(default_factory=dict)


class Adapter:
    name = "base"

    def describe(self) -> dict:
        raise NotImplementedError

    def generate(self, batch: list[list[dict]], cfg: GenConfig) -> list[GenResult]:
        raise NotImplementedError


# ------------------------------------------------------------------ local HF
class HFAdapter(Adapter):
    name = "hf"

    def __init__(self, model: str, revision: str | None = None, adapter_path: str | None = None,
                 dtype: str = "bfloat16", device: str = "auto", threads: int | None = None, load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self.model_id, self.revision, self.adapter_path = model, revision, adapter_path
        self.tok = AutoTokenizer.from_pretrained(model, revision=revision)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        td = getattr(torch, dtype)
        self.load_in_4bit = load_in_4bit
        if load_in_4bit:  # e.g. a 27B baseline on a 40GB GPU (documented as a 4-bit handicap)
            from transformers import BitsAndBytesConfig
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=td)
            self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=td, quantization_config=q,
                                                              device_map={"": 0})
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=td).to(device)
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()
        self.model.eval()
        self.dtype = dtype

    def describe(self) -> dict:
        import torch, transformers
        return {"adapter": "hf", "model": self.model_id, "revision": self.revision, "lora": self.adapter_path,
                "dtype": self.dtype, "device": self.device, "load_in_4bit": self.load_in_4bit, "torch": torch.__version__,
                "transformers": transformers.__version__, "threads": torch.get_num_threads()}

    def render(self, msgs: list[dict], cfg: GenConfig) -> str:
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=cfg.enable_thinking)

    def generate(self, batch, cfg):
        import torch
        prompts = [self.render(m, cfg) for m in batch]
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
        gen_kwargs = dict(max_new_tokens=cfg.max_new_tokens, pad_token_id=self.tok.pad_token_id)
        if cfg.temperature > 0:
            torch.manual_seed(cfg.seed)
            gen_kwargs.update(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p)
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        dt = time.time() - t0
        res = []
        L = enc["input_ids"].shape[1]
        eos_ids = set([self.tok.eos_token_id] + ([self.tok.convert_tokens_to_ids("<|im_end|>")]))
        for i in range(len(batch)):
            gen = out[i, L:]
            n_in = int(enc["attention_mask"][i].sum())
            toks = gen.tolist()
            n_out = 0
            finished = False
            for t in toks:
                if t == self.tok.pad_token_id and finished:
                    break
                n_out += 1
                if t in eos_ids:
                    finished = True
                    break
            text = self.tok.decode(gen[:n_out], skip_special_tokens=True)
            res.append(GenResult(text, n_in, n_out, dt / len(batch), "stop" if finished else "length",
                                 extra={"batch_latency_s": dt, "batch_size": len(batch)}))
        return res


# ------------------------------------------------------------------ OpenAI-compatible
PRICES_PER_MTOK = {
    # Filled from configs/models.yaml at runtime; (input, output) USD per 1M tokens.
}


class OpenAICompatAdapter(Adapter):
    name = "openai"

    def __init__(self, model: str, base_url: str, api_key_env: str = "FINQA_OPENAI_API_KEY",
                 price_in: float | None = None, price_out: float | None = None, timeout: float = 600):
        import httpx
        self.model_id, self.base_url = model, base_url.rstrip("/")
        key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY", "")
        self.client = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {key}"} if key else {})
        self.price_in, self.price_out = price_in, price_out

    def describe(self):
        return {"adapter": "openai_compat", "model": self.model_id, "base_url": self.base_url,
                "price_in_per_mtok": self.price_in, "price_out_per_mtok": self.price_out}

    def _one(self, msgs, cfg):
        body = {"model": self.model_id, "messages": msgs, "max_completion_tokens": cfg.max_new_tokens,
                "temperature": cfg.temperature, "seed": cfg.seed}
        if cfg.reasoning_effort:
            body["reasoning_effort"] = cfg.reasoning_effort
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            body["max_tokens"] = body.pop("max_completion_tokens")
            body["chat_template_kwargs"] = {"enable_thinking": cfg.enable_thinking}
        t0 = time.time()
        r = self.client.post(f"{self.base_url}/chat/completions", json=body)
        r.raise_for_status()
        d = r.json()
        dt = time.time() - t0
        u = d.get("usage", {})
        ti, to = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        cost = None
        if self.price_in is not None:
            cost = ti / 1e6 * self.price_in + to / 1e6 * (self.price_out or 0)
        ch = d["choices"][0]
        return GenResult(ch["message"].get("content") or "", ti, to, dt, ch.get("finish_reason", ""), cost,
                         extra={"usage": u})

    def generate(self, batch, cfg):
        return [self._one(m, cfg) for m in batch]


# ------------------------------------------------------------------ Anthropic
class AnthropicAdapter(Adapter):
    name = "anthropic"

    def __init__(self, model: str, price_in: float | None = None, price_out: float | None = None,
                 base_url: str = "https://api.anthropic.com", timeout: float = 600):
        import httpx
        key = os.environ.get("FINQA_ANTHROPIC_API_KEY", "")
        self.model_id, self.base_url = model, base_url
        self.client = httpx.Client(timeout=timeout, headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
        self.price_in, self.price_out = price_in, price_out

    def describe(self):
        return {"adapter": "anthropic", "model": self.model_id, "base_url": self.base_url,
                "price_in_per_mtok": self.price_in, "price_out_per_mtok": self.price_out}

    def _one(self, msgs, cfg):
        system = "\n".join(m["content"] for m in msgs if m["role"] == "system")
        body = {"model": self.model_id, "max_tokens": cfg.max_new_tokens, "system": system,
                "messages": [m for m in msgs if m["role"] != "system"]}
        if cfg.temperature is not None and not cfg.reasoning_effort:
            body["temperature"] = cfg.temperature
        t0 = time.time()
        r = self.client.post(f"{self.base_url}/v1/messages", json=body)
        r.raise_for_status()
        d = r.json()
        dt = time.time() - t0
        u = d.get("usage", {})
        ti, to = u.get("input_tokens", 0), u.get("output_tokens", 0)
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        cost = None if self.price_in is None else ti / 1e6 * self.price_in + to / 1e6 * (self.price_out or 0)
        return GenResult(text, ti, to, dt, d.get("stop_reason", ""), cost, extra={"usage": u})

    def generate(self, batch, cfg):
        return [self._one(m, cfg) for m in batch]


# ------------------------------------------------------------------ Gemini
class GeminiAdapter(Adapter):
    name = "gemini"

    def __init__(self, model: str, price_in: float | None = None, price_out: float | None = None, timeout: float = 600):
        import httpx
        self.key = os.environ.get("FINQA_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
        self.model_id = model
        self.client = httpx.Client(timeout=timeout)
        self.price_in, self.price_out = price_in, price_out

    def describe(self):
        return {"adapter": "gemini", "model": self.model_id,
                "price_in_per_mtok": self.price_in, "price_out_per_mtok": self.price_out}

    def _one(self, msgs, cfg):
        system = "\n".join(m["content"] for m in msgs if m["role"] == "system")
        contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
                    for m in msgs if m["role"] != "system"]
        gc = {"maxOutputTokens": cfg.max_new_tokens, "temperature": cfg.temperature, "seed": cfg.seed}
        body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents, "generationConfig": gc}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent"
        t0 = time.time()
        r = self.client.post(url, json=body, headers={"x-goog-api-key": self.key})
        r.raise_for_status()
        d = r.json()
        dt = time.time() - t0
        u = d.get("usageMetadata", {})
        ti = u.get("promptTokenCount", 0)
        to = u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
        cand = (d.get("candidates") or [{}])[0]
        text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
        cost = None if self.price_in is None else ti / 1e6 * self.price_in + to / 1e6 * (self.price_out or 0)
        return GenResult(text, ti, to, dt, cand.get("finishReason", ""), cost, extra={"usage": u})

    def generate(self, batch, cfg):
        return [self._one(m, cfg) for m in batch]


def make_adapter(spec: dict) -> Adapter:
    kind = spec["adapter"]
    if kind == "hf":
        return HFAdapter(spec["model"], spec.get("revision"), spec.get("lora"), spec.get("dtype", "bfloat16"),
                         spec.get("device", "auto"), spec.get("threads"), spec.get("load_in_4bit", False))
    if kind == "openai_compat":
        return OpenAICompatAdapter(spec["model"], spec["base_url"], spec.get("api_key_env", "FINQA_OPENAI_API_KEY"),
                                   spec.get("price_in"), spec.get("price_out"))
    if kind == "anthropic":
        return AnthropicAdapter(spec["model"], spec.get("price_in"), spec.get("price_out"))
    if kind == "gemini":
        return GeminiAdapter(spec["model"], spec.get("price_in"), spec.get("price_out"))
    raise ValueError(f"unknown adapter {kind}")
