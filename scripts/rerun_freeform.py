"""Re-run free-form schema probe with correct Gemini content parsing."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
load_dotenv(ROOT / ".env")

from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
from backend.schemas.routing import RouterIntentSchema  # noqa: E402

RESULTS = ROOT / "report" / "results"
FIGURES = ROOT / "report" / "figures"


def content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and part.get("text"):
                    chunks.append(str(part["text"]))
                elif "text" in part and part.get("type") != "thinking":
                    chunks.append(str(part["text"]))
            else:
                text = getattr(part, "text", None)
                ptype = getattr(part, "type", None)
                if text and ptype != "thinking":
                    chunks.append(str(text))
        return "\n".join(chunks)
    return str(content)


def try_parse_schema(text: str) -> bool:
    text = (text or "").strip()
    if "```" in text:
        chunk = text.split("```", 2)[1]
        if chunk.lstrip().startswith("json"):
            chunk = chunk.lstrip()[4:]
        text = chunk.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return False
    blob = text[start : end + 1]
    try:
        RouterIntentSchema.model_validate_json(blob)
        return True
    except Exception:
        try:
            data = json.loads(blob)
            RouterIntentSchema.model_validate(data)
            return True
        except Exception:
            return False


async def main() -> None:
    model = os.getenv("GEMINI_MODEL", "gemma-4-31b-it")
    api_key = os.getenv("GOOGLE_API_KEY")
    system = (
        "Sei un router per un assistente ETF. "
        "Rispondi SOLO con un JSON valido, senza markdown e senza testo extra, "
        'con chiavi: intent (etf|chit_chat|out_of_domain), is_routable (bool), '
        "direct_response, clean_query, isin."
    )
    probes = [
        "Analizza IE00B4L5Y983",
        "Ciao",
        "Che tempo fa a Roma?",
        "Cosa sono gli ETF?",
        "Buongiorno, come va?",
        "Scrivi codice Python per un server HTTP",
        "ISIN IE00B3XXRP09 holdings",
        "Raccontami una barzelletta",
        "Mi consigli un ETF ESG europeo?",
        "Grazie mille",
        "Mostra Sharpe e beta di DE000A0F5UF5",
        "Apri Spotify",
    ]
    temps = [0.0, 0.3, 0.7, 1.0]
    free_by_temp = {}
    samples = []

    for t in temps:
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=t,
            timeout=45.0,
            max_retries=1,
            model_kwargs={"system_instruction": system},
        )
        oks = []
        for q in probes:
            try:
                resp = await llm.ainvoke(q)
                text = content_to_text(resp.content)
                ok = try_parse_schema(text)
            except Exception as exc:
                text = f"ERROR:{exc}"
                ok = False
            oks.append(ok)
            samples.append({"temperature": t, "query": q, "schema_ok": ok, "preview": text[:180]})
            print(f"T={t} ok={ok} q={q!r} preview={text[:80]!r}", flush=True)
            await asyncio.sleep(0.05)
        free_by_temp[str(t)] = {"n": len(oks), "schema_pass": sum(oks) / len(oks)}

    summary_path = RESULTS / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["llm"]["by_temperature_freeform"] = free_by_temp
    summary["llm"]["freeform_note"] = (
        "Free-form uses plain generation (no with_structured_output); "
        "parser reads only text parts (ignores thinking)."
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # rewrite ablation figure
    temps_f = temps
    struct = [
        summary["llm"]["by_temperature_structured"][str(t)]["schema_pass"] for t in temps_f
    ]
    intent = [
        summary["llm"]["by_temperature_structured"][str(t)]["intent_accuracy"]
        for t in temps_f
    ]
    free = [free_by_temp[str(t)]["schema_pass"] for t in temps_f]
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(temps_f, struct, "o-", label="Schema pass (structured output)", linewidth=2)
    plt.plot(temps_f, intent, "s--", label="Intent accuracy (structured)", linewidth=2)
    plt.plot(temps_f, free, "^-", label="Schema pass (free-form JSON)", linewidth=2)
    plt.xlabel("Temperatura T")
    plt.ylabel("Tasso")
    plt.ylim(-0.05, 1.05)
    plt.title("Ablation temperatura: structured vs free-form")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "temperature_ablation.png", dpi=180)
    plt.close()

    (RESULTS / "llm_freeform_cases.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Updated freeform:", free_by_temp)


if __name__ == "__main__":
    asyncio.run(main())
