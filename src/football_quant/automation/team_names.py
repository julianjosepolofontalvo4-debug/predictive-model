from __future__ import annotations

import re
import unicodedata


# Canonical names are stable internal names. Aliases are deliberately broad
# enough for common API/manual variations but never fuzzy-match silently.
TEAM_ALIASES = {
    "villarreal": "Villarreal CF",
    "villarreal cf": "Villarreal CF",
    "real betis": "Real Betis Balompié",
    "real betis cf": "Real Betis Balompié",
    "real betis balompie": "Real Betis Balompié",
    "real betis seville": "Real Betis Balompié",
    "betis": "Real Betis Balompié",
    "atletico madrid": "Club Atlético de Madrid",
    "club atletico de madrid": "Club Atlético de Madrid",
    "atletico de madrid": "Club Atlético de Madrid",
    "athletic bilbao": "Athletic Club",
    "athletic club": "Athletic Club",
    "deportivo alaves": "Deportivo Alavés",
    "alaves": "Deportivo Alavés",
    "rc celta de vigo": "Real Club Celta de Vigo",
    "celta vigo": "Real Club Celta de Vigo",
    "celta": "Real Club Celta de Vigo",
    "ca osasuna": "Club Atlético Osasuna",
    "osasuna": "Club Atlético Osasuna",
    "malaga": "Málaga CF",
    "malaga cf": "Málaga CF",
    "espanyol barcelona": "RCD Espanyol",
    "espanyol": "RCD Espanyol",
    "deportivo": "RC Deportivo de A Coruña",
    "deportivo de la coruna": "RC Deportivo de A Coruña",
    "racing santander": "Real Racing Club",
    "real racing club": "Real Racing Club",
    "real sociedad san sebastian": "Real Sociedad",
    "real sociedad": "Real Sociedad",
    "barcelona": "FC Barcelona",
    "fc barcelona": "FC Barcelona",
    "manchester united": "Manchester United",
    "man utd": "Manchester United",
    "manchester city": "Manchester City",
    "man city": "Manchester City",
    "tottenham": "Tottenham Hotspur",
    "tottenham hotspur": "Tottenham Hotspur",
    "west ham": "West Ham United",
    "newcastle": "Newcastle United",
    "newcastle united": "Newcastle United",
    "wolves": "Wolverhampton Wanderers",
    "wolverhampton": "Wolverhampton Wanderers",
    "nottingham forest": "Nottingham Forest",
    "brighton": "Brighton & Hove Albion",
    "brighton and hove albion": "Brighton & Hove Albion",
    "psg": "Paris Saint-Germain",
    "paris sg": "Paris Saint-Germain",
    "inter": "Inter Milan",
    "internazionale": "Inter Milan",
    "ac milan": "AC Milan",
    "milan": "AC Milan",
}


def _key(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold().strip()
    text = re.sub(r"[^\w\s&-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_team_name(name: str) -> str:
    """Canonicalize a known team alias; unknown names remain unchanged."""
    clean = str(name).strip()
    if not clean:
        raise ValueError("El nombre de equipo no puede estar vacío.")
    return TEAM_ALIASES.get(_key(clean), clean)
