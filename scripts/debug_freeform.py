from dotenv import load_dotenv
load_dotenv(".env")
import os, sys, asyncio, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.schemas.routing import RouterIntentSchema

async def main():
    prompt = (
        'Rispondi SOLO con un JSON che rispetta: '
        '{"intent":"etf|chit_chat|out_of_domain","is_routable":bool,'
        '"direct_response":str|null,"clean_query":str|null,"isin":str|null}'
    )
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0,
        timeout=45.0,
        max_retries=1,
        model_kwargs={"system_instruction": prompt},
    )
    for q in ["Ciao", "Analizza IE00B4L5Y983", "Che tempo fa a Roma?"]:
        r = await llm.ainvoke(q)
        content = r.content
        print("=" * 60)
        print("Q:", q)
        print("type:", type(content))
        print("repr:", repr(content)[:1000])
        text = content
        if isinstance(content, list):
            text = "".join(getattr(p, "text", str(p)) for p in content)
        text = str(text).strip()
        if "```" in text:
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        start, end = text.find("{"), text.rfind("}")
        print("slice:", text[start : end + 1] if start >= 0 else None)
        if start >= 0 and end > start:
            try:
                obj = RouterIntentSchema.model_validate_json(text[start : end + 1])
                print("VALID", obj)
            except Exception as e:
                print("INVALID", e)
                try:
                    print("json.loads", json.loads(text[start : end + 1]))
                except Exception as e2:
                    print("json.loads fail", e2)

asyncio.run(main())
