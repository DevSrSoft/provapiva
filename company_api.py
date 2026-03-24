"""
Company Info API - FastAPI v2.1
- VIES EU → validazione P.IVA + ragione sociale + indirizzo
- Gemini (Google Search grounding) → PEC, codice ATECO, telefono, sito web, ecc.

Installazione:
    pip install fastapi uvicorn httpx google-genai

Configurazione:
    export GEMINI_API_KEY="..."

Avvio:
    uvicorn company_api:app --reload

Endpoint:
    GET /company/{partita_iva}          → dati base VIES
    GET /company/{partita_iva}/extra    → dati base + dati extra via Gemini
    GET /health
"""

import json
import os
import re
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from google import genai
from google.genai import types

app = FastAPI(
    title="Company Info API",
    description="Dati aziendali italiani da P.IVA — VIES EU + Gemini Google Search",
    version="2.1.0",
)

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ─── Modelli ────────────────────────────────────────────────────────────────

class Address(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    zip_code: Optional[str] = None
    country: Optional[str] = None


class CompanyInfo(BaseModel):
    partita_iva: str
    valid: bool
    company_name: Optional[str] = None
    address: Optional[Address] = None
    country_code: str = "IT"
    vies_status: str
    raw_vies: Optional[dict] = None


class CompanyInfoExtra(CompanyInfo):
    pec: Optional[str] = None
    telefono: Optional[str] = None
    sito_web: Optional[str] = None
    codice_ateco: Optional[str] = None
    descrizione_ateco: Optional[str] = None
    forma_giuridica: Optional[str] = None
    codice_sdi: Optional[str] = None
    ai_source: Optional[str] = None
    ai_note: Optional[str] = None


# ─── Helpers ────────────────────────────────────────────────────────────────

def clean_piva(piva: str) -> str:
    piva = piva.strip().upper().replace(" ", "")
    if piva.startswith("IT"):
        piva = piva[2:]
    return piva


def validate_piva_format(piva: str) -> bool:
    return bool(re.fullmatch(r"\d{11}", piva))


def parse_address(raw: Optional[str]) -> Address:
    if not raw:
        return Address()
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    street = lines[0] if lines else None
    city = lines[1] if len(lines) > 1 else None
    zip_code = None
    if city:
        m = re.search(r"\b(\d{5})\b", city)
        if m:
            zip_code = m.group(1)
            city = city.replace(zip_code, "").strip()
    return Address(street=street, city=city, zip_code=zip_code, country="Italia")


def normalize(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    val = val.strip()
    return None if val in ("---", "", "N/A", "-") else val


def _validate_pec(pec: Optional[str]) -> Optional[str]:
    """Valida che sia un indirizzo PEC reale."""
    if not pec:
        return None
    pec = pec.strip().lower()
    if not re.fullmatch(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", pec):
        return None
    if "protected" in pec or "example" in pec:
        return None
    return pec


# ─── VIES ───────────────────────────────────────────────────────────────────

async def fetch_vies(piva: str) -> dict:
    url = f"https://ec.europa.eu/taxation_customs/vies/rest-api/ms/IT/vat/{piva}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(502, f"VIES errore: {e.response.status_code}")
        except httpx.RequestError:
            raise HTTPException(503, "VIES non raggiungibile. Riprova tra poco.")


def build_company_info(piva: str, raw: dict) -> dict:
    is_valid = raw.get("isValid", False)
    company_name = normalize(raw.get("name") or raw.get("traderName"))
    raw_address = normalize(raw.get("address") or raw.get("traderAddress"))
    return dict(
        partita_iva=piva,
        valid=is_valid,
        company_name=company_name,
        address=parse_address(raw_address),
        country_code="IT",
        vies_status="valid" if is_valid else "invalid",
        raw_vies=raw,
    )


# ─── Gemini Google Search ──────────────────────────────────────────────────

def _gemini_search(prompt: str) -> str:
    """Chiama Gemini con Google Search grounding e restituisce il testo."""
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.1,
        ),
    )
    # gemini-2.5 può avere response.text = None quando usa thinking
    # In quel caso estraiamo il testo dalle parti della risposta
    if response.text:
        return response.text
    try:
        for candidate in response.candidates:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.text and not part.thought:
                        return part.text
    except Exception:
        pass
    return ""


async def search_company_data(piva: str, company_name: Optional[str]) -> dict:
    """
    Usa Gemini con Google Search grounding per trovare PEC, ATECO, telefono, ecc.
    """
    nome = company_name or ""

    prompt = f"""Cerca informazioni sull'azienda italiana "{nome}" con Partita IVA {piva}.

CERCA SUL WEB tutti questi dati:

1. PEC (Posta Elettronica Certificata): cerca su registroimprese.it, inipec.gov.it, dnb.com, europages.it, kompass.com, tuttitalia.it, informazione-aziende.it, o sul sito ufficiale dell'azienda (contatti, footer, privacy policy, PDF pubblici).
   La PEC ha domini come: @pec.it, @legalmail.it, @pec.aruba.it, @cert.legalmail.it, @pecveneto.it, @pec.buffetti.it, @postacert.it

2. Telefono principale
3. Sito web ufficiale
4. Codice ATECO e descrizione attività
5. Forma giuridica (SRL, SPA, ecc.)
6. Codice SDI (codice destinatario per fatturazione elettronica)

Restituisci SOLO un JSON valido, niente altro testo:
{{
    "pec": "indirizzo PEC reale oppure null",
    "telefono": "numero oppure null",
    "sito_web": "URL oppure null",
    "codice_ateco": "codice oppure null",
    "descrizione_ateco": "descrizione oppure null",
    "forma_giuridica": "tipo oppure null",
    "codice_sdi": "codice oppure null"
}}

REGOLE:
- SOLO JSON valido, niente altro
- La PEC deve essere un indirizzo email reale (con @ e dominio), NON email normali
- Se trovi "[email protected]" o testo offuscato, metti null per la PEC
- NON inventare dati — solo dati reali trovati sul web
- Se non trovi un dato, metti null"""

    try:
        result_text = _gemini_search(prompt)
        if not result_text:
            return _empty_result("Nessuna risposta da Gemini")

        # Pulisci il JSON
        result_text = result_text.strip()
        if result_text.startswith("```"):
            result_text = re.sub(r"^```\w*\n?", "", result_text)
            result_text = re.sub(r"\n?```$", "", result_text)
        result_text = result_text.strip()

        data = json.loads(result_text)

        # Normalizza: Gemini a volte usa chiavi diverse o "Non disponibile"
        def get_val(data: dict, *keys: str) -> Optional[str]:
            for k in keys:
                v = data.get(k)
                if v and str(v).strip().lower() not in (
                    "non disponibile", "n/a", "null", "none", "-", "---",
                ):
                    return str(v).strip()
            return None

        # Gestisci codice ATECO che può includere la descrizione
        ateco_raw = get_val(data, "codice_ateco", "Codice ATECO")
        codice_ateco = ateco_raw
        descrizione_ateco = get_val(data, "descrizione_ateco", "Descrizione ATECO")
        if ateco_raw and " - " in ateco_raw:
            parts = ateco_raw.split(" - ", 1)
            codice_ateco = parts[0].strip()
            if not descrizione_ateco:
                descrizione_ateco = parts[1].strip()

        return {
            "pec": _validate_pec(get_val(data, "pec", "PEC")),
            "telefono": get_val(data, "telefono", "Telefono"),
            "sito_web": get_val(data, "sito_web", "Sito Web", "sito web"),
            "codice_ateco": codice_ateco,
            "descrizione_ateco": descrizione_ateco,
            "forma_giuridica": get_val(data, "forma_giuridica", "Forma Giuridica"),
            "codice_sdi": get_val(data, "codice_sdi", "Codice SDI"),
            "ai_source": "Gemini 2.5 Flash + Google Search",
            "ai_note": None,
        }

    except json.JSONDecodeError:
        return _empty_result(f"Risposta Gemini non parsabile: {result_text[:200]}")
    except Exception as e:
        return _empty_result(f"Errore Gemini: {str(e)}")


def _empty_result(note: str) -> dict:
    return {
        "pec": None,
        "telefono": None,
        "sito_web": None,
        "codice_ateco": None,
        "descrizione_ateco": None,
        "forma_giuridica": None,
        "codice_sdi": None,
        "ai_source": None,
        "ai_note": note,
    }


# ─── Endpoint ────────────────────────────────────────────────────────────────

@app.get("/company/{partita_iva}", response_model=CompanyInfo, tags=["Company"])
async def get_company(partita_iva: str):
    """Dati base da VIES (ragione sociale, indirizzo, validità P.IVA)."""
    piva = clean_piva(partita_iva)
    if not validate_piva_format(piva):
        raise HTTPException(422, f"P.IVA non valida: '{piva}'. Servono 11 cifre.")
    raw = await fetch_vies(piva)
    return CompanyInfo(**build_company_info(piva, raw))


@app.get(
    "/company/{partita_iva}/extra",
    response_model=CompanyInfoExtra,
    tags=["Company"],
)
async def get_company_extra(partita_iva: str):
    """
    Dati base VIES + dati extra (PEC, ATECO, telefono, sito web, ecc.)
    trovati tramite Gemini con Google Search grounding.

    Richiede GEMINI_API_KEY configurata.
    """
    piva = clean_piva(partita_iva)
    if not validate_piva_format(piva):
        raise HTTPException(422, f"P.IVA non valida: '{piva}'. Servono 11 cifre.")

    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            500,
            "GEMINI_API_KEY non configurata. Esegui: export GEMINI_API_KEY='...'",
        )

    # VIES prima (serve il nome per la ricerca Gemini)
    raw = await fetch_vies(piva)
    base = build_company_info(piva, raw)

    # Gemini + Google Search
    extra = await search_company_data(piva, base.get("company_name"))

    return CompanyInfoExtra(**base, **extra)


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "version": "2.1.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("company_api:app", host="0.0.0.0", port=8000, reload=True)
