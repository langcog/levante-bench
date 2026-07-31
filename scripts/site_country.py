"""Map LEVANTE pilot `site` codes to ISO-ish country keys used in persona profiles.

Shared by build_age_accuracy_profile.py and build_age_ability_profile.py.
Unknown / NA / blank sites are omitted from country-stratified tables (they still
contribute to the global age profiles).
"""
from __future__ import annotations

# Pilot site id -> country key (lowercase ISO 3166-1 alpha-2 where unambiguous).
SITE_TO_COUNTRY: dict[str, str] = {
    "pilot_mpieva_de": "de",
    "pilot_uniandes_co": "co",
    "pilot_western_ca": "ca",
}


def site_to_country(site: str | None) -> str | None:
    if not site:
        return None
    s = str(site).strip()
    if not s or s.upper() == "NA":
        return None
    return SITE_TO_COUNTRY.get(s)
