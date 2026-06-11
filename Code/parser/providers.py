import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Protocol

from dotenv import load_dotenv
from pydantic import ValidationError

from .schemas import JDRequirements, ResumeData


load_dotenv()


class _SlidingRateLimiter:
    def __init__(self, rpm: int, rpd: int, tpm: int, tpd: int):
        self.rpm, self.rpd, self.tpm, self.tpd = rpm, rpd, tpm, tpd
        self._req_min: deque = deque()
        self._req_day: deque = deque()
        self._tok_min: deque = deque()
        self._tok_day: deque = deque()
        self._lock = Lock()

    def _purge(self, now: float):
        while self._req_min and now - self._req_min[0] >= 60:
            self._req_min.popleft()
        while self._req_day and now - self._req_day[0] >= 86400:
            self._req_day.popleft()
        while self._tok_min and now - self._tok_min[0][0] >= 60:
            self._tok_min.popleft()
        while self._tok_day and now - self._tok_day[0][0] >= 86400:
            self._tok_day.popleft()

    @staticmethod
    def _sum_tokens(q: deque) -> int:
        return sum(t for _, t in q)

    def acquire(self, estimated_tokens: int):
        while True:
            with self._lock:
                now = time.monotonic()
                self._purge(now)
                wait = 0.0
                if len(self._req_min) >= self.rpm:
                    wait = max(wait, 60 - (now - self._req_min[0]))
                if len(self._req_day) >= self.rpd:
                    wait = max(wait, 86400 - (now - self._req_day[0]))
                if self._sum_tokens(self._tok_min) + estimated_tokens > self.tpm and self._tok_min:
                    wait = max(wait, 60 - (now - self._tok_min[0][0]))
                if self._sum_tokens(self._tok_day) + estimated_tokens > self.tpd and self._tok_day:
                    wait = max(wait, 86400 - (now - self._tok_day[0][0]))
                if wait <= 0:
                    self._req_min.append(now)
                    self._req_day.append(now)
                    return
            time.sleep(min(wait + 0.1, 10))

    def record_tokens(self, tokens: int):
        with self._lock:
            now = time.monotonic()
            self._tok_min.append((now, tokens))
            self._tok_day.append((now, tokens))


class LLMProvider(Protocol):
    def extract_resume(self, prompt: str) -> ResumeData:
        ...
    def extract_jd(self, prompt: str) -> JDRequirements:
        ...


_RESUME_TOP_KEYS = {"candidate", "summary", "skills", "education", "experience", "parser_metadata"}
_JD_TOP_KEYS = {"required_skills", "preferred_skills", "required_qualifications", "min_years_experience"}


def _unwrap_response(payload, expected_keys):
    if not isinstance(payload, dict):
        return payload
    if expected_keys & payload.keys():
        return payload
    if len(payload) == 1:
        inner = next(iter(payload.values()))
        if isinstance(inner, dict) and (expected_keys & inner.keys()):
            return inner
    return payload


def _retry_validate(call_once, provider_label: str, prompt: str, model_cls=None, expected_keys=None):
    if model_cls is None:
        model_cls = ResumeData
    if expected_keys is None:
        expected_keys = _RESUME_TOP_KEYS

    validation_error = None
    validation_prompt = prompt

    for attempt in range(2):
        response_text = call_once(validation_prompt).strip()

        try:
            payload = _unwrap_response(json.loads(response_text), expected_keys)
            return model_cls.model_validate(payload)
        except (ValidationError, json.JSONDecodeError) as exc:
            validation_error = exc

            if attempt == 0:
                validation_prompt = (
                    "The previous response was invalid or did not match the required schema. "
                    "Return only valid JSON matching the schema exactly — no markdown, no commentary, "
                    "no wrapping keys.\n\n"
                    f"Original prompt:\n{prompt}"
                )

    raise ValueError(
        f"{provider_label} returned an invalid structured response: {validation_error}"
    )


class OpenRouterProvider:
    def __init__(self):
        self.model_name = os.getenv("OPENROUTER_MODEL") or "openrouter/owl-alpha"
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_url = os.getenv("OPENROUTER_API_URL") or "https://openrouter.ai/api/v1/chat/completions"
        self.timeout_seconds = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS") or 180)
        self.max_tokens = int(os.getenv("OPENROUTER_MAX_TOKENS") or 4096)
        self.http_referer = os.getenv("OPENROUTER_HTTP_REFERER")
        self.app_title = os.getenv("OPENROUTER_APP_TITLE")

    def _headers(self):
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer

        if self.app_title:
            headers["X-OpenRouter-Title"] = self.app_title

        return headers

    def _payload(self, prompt: str, schema_cls=None, schema_name: str = "resume_extraction"):
        if schema_cls is None:
            schema_cls = ResumeData
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema_cls.model_json_schema()
                }
            },
            "temperature": 0,
            "max_tokens": self.max_tokens
        }

    def _post(self, prompt: str, schema_cls=None, schema_name: str = "resume_extraction") -> str:
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(self._payload(prompt, schema_cls, schema_name)).encode("utf-8"),
            headers=self._headers(),
            method="POST"
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_json = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenRouter request failed with HTTP {exc.code}: {error_body}"
            ) from exc

        choices = response_json.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content") or ""

    def extract_resume(self, prompt: str) -> ResumeData:
        return _retry_validate(self._post, "OpenRouter", prompt)

    def extract_jd(self, prompt: str) -> JDRequirements:
        return _retry_validate(
            lambda p: self._post(p, JDRequirements, "jd_extraction"),
            "OpenRouter", prompt, JDRequirements, _JD_TOP_KEYS
        )


class GeminiProvider:
    def __init__(self):
        from google import genai
        from google.genai import types

        self._types = types
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        self.client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

    def _call(self, prompt: str, schema_cls=None) -> str:
        if schema_cls is None:
            schema_cls = ResumeData
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema_cls,
                temperature=0
            )
        )
        return response.text or ""

    def extract_resume(self, prompt: str) -> ResumeData:
        return _retry_validate(self._call, "Gemini", prompt)

    def extract_jd(self, prompt: str) -> JDRequirements:
        return _retry_validate(
            lambda p: self._call(p, JDRequirements),
            "Gemini", prompt, JDRequirements, _JD_TOP_KEYS
        )


class OllamaProvider:
    def __init__(self):
        from ollama import Client

        self.model_name = os.getenv("OLLAMA_MODEL") or "gpt-oss:120b-cloud"
        self.host = os.getenv("OLLAMA_HOST") or None
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX") or 8192)
        self.num_predict = int(os.getenv("OLLAMA_NUM_PREDICT") or 4096)
        self.client = Client(host=self.host) if self.host else Client()

    def _call(self, prompt: str, schema_cls=None) -> str:
        if schema_cls is None:
            schema_cls = ResumeData
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            format=schema_cls.model_json_schema(),
            options={
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "temperature": 0
            }
        )
        message = getattr(response, "message", None)
        if not message:
            return ""
        return message.content or ""

    def extract_resume(self, prompt: str) -> ResumeData:
        return _retry_validate(self._call, "Ollama", prompt)

    def extract_jd(self, prompt: str) -> JDRequirements:
        return _retry_validate(
            lambda p: self._call(p, JDRequirements),
            "Ollama", prompt, JDRequirements, _JD_TOP_KEYS
        )


class GroqProvider:
    def __init__(self):
        self.model_name = os.getenv("GROQ_MODEL") or "openai/gpt-oss-20b"
        self.api_key = os.getenv("GROQ_API_KEY")
        self.api_url = os.getenv("GROQ_API_URL") or "https://api.groq.com/openai/v1/chat/completions"
        self.timeout_seconds = float(os.getenv("GROQ_TIMEOUT_SECONDS") or 180)
        self.max_tokens = int(os.getenv("GROQ_MAX_TOKENS") or 4096)
        self.limiter = _SlidingRateLimiter(
            rpm=int(os.getenv("GROQ_RPM") or 30),
            rpd=int(os.getenv("GROQ_RPD") or 1000),
            tpm=int(os.getenv("GROQ_TPM") or 8000),
            tpd=int(os.getenv("GROQ_TPD") or 200000),
        )

    def _headers(self):
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "resume-parser/1.0"
        }

    def _budget_max_tokens(self, prompt: str) -> int:
        estimated_input = max(1, len(prompt) // 4)
        headroom = self.limiter.tpm - estimated_input - 200
        if headroom < 256:
            headroom = 256
        return max(256, min(self.max_tokens, headroom))

    def _payload(self, prompt: str):
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": self._budget_max_tokens(prompt)
        }
        reasoning_effort = os.getenv("GROQ_REASONING_EFFORT") or "low"
        if reasoning_effort and reasoning_effort.lower() != "none":
            payload["reasoning_effort"] = reasoning_effort.lower()
        return payload

    @staticmethod
    def _estimate_tokens(prompt: str, max_out: int) -> int:
        # rough ~4 chars/token + reserved completion budget
        return max(1, len(prompt) // 4) + max_out

    def _post(self, prompt: str) -> str:
        self.limiter.acquire(self._estimate_tokens(prompt, self.max_tokens))

        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(self._payload(prompt)).encode("utf-8"),
            headers=self._headers(),
            method="POST"
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_json = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 30.0
                except ValueError:
                    delay = 30.0
                time.sleep(min(delay + 0.5, 120))
                return self._post(prompt)
            try:
                parsed = json.loads(body)
                failed_generation = (parsed.get("error") or {}).get("failed_generation") or ""
                if failed_generation:
                    print(f"[Groq] failed_generation preview ({len(failed_generation)} chars):\n{failed_generation[:1500]}", flush=True)
            except Exception:
                pass
            raise RuntimeError(f"Groq request failed with HTTP {exc.code}: {body}") from exc

        usage = response_json.get("usage") or {}
        total = int(usage.get("total_tokens") or 0)
        if total:
            self.limiter.record_tokens(total)
        else:
            self.limiter.record_tokens(self._estimate_tokens(prompt, self.max_tokens))

        choices = response_json.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content") or ""

    def extract_resume(self, prompt: str) -> ResumeData:
        return _retry_validate(self._post, "Groq", prompt)

    def extract_jd(self, prompt: str) -> JDRequirements:
        return _retry_validate(self._post, "Groq", prompt, JDRequirements, _JD_TOP_KEYS)


def get_provider() -> LLMProvider:
    name = (os.getenv("LLM_PROVIDER") or "openrouter").strip().lower()

    if name == "gemini":
        return GeminiProvider()
    if name == "ollama":
        return OllamaProvider()
    if name == "openrouter":
        return OpenRouterProvider()
    if name == "groq":
        return GroqProvider()

    raise ValueError(f"Unknown LLM_PROVIDER: {name!r} (expected gemini, groq, ollama, or openrouter)")


def load_or_extract_jd(
    jd_file_path: str,
    provider: LLMProvider,
    archive_dir: Path | None = None,
    cache_dir: Path = Path("JDCache"),
) -> JDRequirements:
    from .extract import extract_document_text_and_links
    from .prompts import build_jd_extraction_prompt

    path = Path(jd_file_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (path.name + ".jdcache")

    raw = path.read_bytes()
    file_hash = hashlib.sha256(raw).hexdigest()

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("hash") == file_hash:
                print(f"[JD cache] hit: {path.name}", flush=True)
                return JDRequirements.model_validate(cached["requirements"])
            raise RuntimeError(
                f"\n[JD cache] conflict: '{path.name}' has different content from the cached version.\n"
                f"  → Rename the new file to a different name and re-run, or\n"
                f"  → Delete '{cache_path}' to replace the cached JD."
            )
        except RuntimeError:
            raise
        except Exception:
            pass  # corrupt cache — fall through to re-extract

    print(f"[JD cache] miss: {path.name} — calling LLM", flush=True)
    text, _links = extract_document_text_and_links(path)
    prompt = build_jd_extraction_prompt(text)
    requirements = provider.extract_jd(prompt)

    try:
        cache_path.write_text(
            json.dumps({"hash": file_hash, "requirements": requirements.model_dump()}, indent=2),
            encoding="utf-8"
        )
        print(f"[JD cache] written: {cache_path}", flush=True)
    except Exception as exc:
        print(f"[JD cache] warning: could not write cache: {exc}", flush=True)

    if archive_dir is not None:
        import shutil
        try:
            archive_dir.mkdir(parents=True, exist_ok=True)
            dest = archive_dir / path.name
            shutil.copy2(path, dest)
            print(f"[JD archive] copied to: {dest}", flush=True)
        except Exception as exc:
            print(f"[JD archive] warning: could not archive JD: {exc}", flush=True)

    return requirements
