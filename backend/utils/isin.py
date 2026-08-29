"""
Formal ISO 6166 ISIN Validation Module.

Implements the Pure Fabrication and Information Expert design patterns (GRASP / GoF)
for deterministic mathematical validation of International Securities Identification Numbers (ISIN)
using the official ISO 6166 Luhn Modulo 36 (Double-Add-Double) checksum algorithm.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

ISIN_SYNTAX_PATTERN = re.compile(r"\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b", re.IGNORECASE)

# ISO 3166-1 alpha-2 country codes + special international prefixes (XS for Euroclear/Clearstream, EU, QS)
VALID_COUNTRY_CODES = {
    "AD", "AE", "AF", "AG", "AI", "AL", "AM", "AO", "AQ", "AR", "AS", "AT", "AU", "AW", "AX", "AZ",
    "BA", "BB", "BD", "BE", "BF", "BG", "BH", "BI", "BJ", "BL", "BM", "BN", "BO", "BQ", "BR", "BS",
    "BT", "BV", "BW", "BY", "BZ", "CA", "CC", "CD", "CF", "CG", "CH", "CI", "CK", "CL", "CM", "CN",
    "CO", "CR", "CU", "CV", "CW", "CX", "CY", "CZ", "DE", "DJ", "DK", "DM", "DO", "DZ", "EC", "EE",
    "EG", "EH", "ER", "ES", "ET", "EU", "FI", "FJ", "FK", "FM", "FO", "FR", "GA", "GB", "GD", "GE",
    "GF", "GG", "GH", "GI", "GL", "GM", "GN", "GP", "GQ", "GR", "GS", "GT", "GU", "GW", "GY", "HK",
    "HM", "HN", "HR", "HT", "HU", "ID", "IE", "IL", "IM", "IN", "IO", "IQ", "IR", "IS", "IT", "JE",
    "JM", "JO", "JP", "KE", "KG", "KH", "KI", "KM", "KN", "KP", "KR", "KW", "KY", "KZ", "LA", "LB",
    "LC", "LI", "LK", "LR", "LS", "LT", "LU", "LV", "LY", "MA", "MC", "MD", "ME", "MF", "MG", "MH",
    "MK", "ML", "MM", "MN", "MO", "MP", "MQ", "MR", "MS", "MT", "MU", "MV", "MW", "MX", "MY", "MZ",
    "NA", "NC", "NE", "NF", "NG", "NI", "NL", "NO", "NP", "NR", "NU", "NZ", "OM", "PA", "PE", "PF",
    "PG", "PH", "PK", "PL", "PM", "PN", "PR", "PS", "PT", "PW", "PY", "QA", "QS", "RE", "RO", "RS",
    "RU", "RW", "SA", "SB", "SC", "SD", "SE", "SG", "SH", "SI", "SJ", "SK", "SL", "SM", "SN", "SO",
    "SR", "SS", "ST", "SV", "SX", "SY", "SZ", "TC", "TD", "TF", "TG", "TH", "TJ", "TK", "TL", "TM",
    "TN", "TO", "TR", "TT", "TV", "TW", "TZ", "UA", "UG", "UM", "US", "UY", "UZ", "VA", "VC", "VE",
    "VG", "VI", "VN", "VU", "WF", "WS", "XC", "XL", "XS", "YE", "YT", "ZA", "ZM", "ZW"
}


class ISINValidationResult(BaseModel):
    """Detailed audit report of an ISIN validation attempt."""

    isin: str = Field(description="Codice ISIN analizzato")
    is_valid: bool = Field(description="Esito complessivo della validazione formale ISO 6166")
    country_code: str = Field(description="Prefisso paese ISO 3166-1 alpha-2")
    is_country_valid: bool = Field(description="Validità del codice paese geografico/emittente")
    expected_check_digit: int = Field(description="Check digit calcolato con formula di Luhn mod-36")
    actual_check_digit: int = Field(description="Check digit presente nel 12° carattere")
    is_check_digit_valid: bool = Field(description="Corrispondenza tra check digit atteso ed effettivo")
    numerical_expansion: str = Field(description="Espansione numerica base-36 prima del checksum")
    error_message: Optional[str] = Field(default=None, description="Motivazione in caso di invalidità")


class ISINValidator:
    """
    Pure Fabrication / Information Expert for ISO 6166 International Securities Identification Numbers.
    Applies the official Double-Add-Double (Luhn Modulo 36) checksum algorithm.
    """

    @staticmethod
    def _char_to_digits(char: str) -> str:
        """Converts alphanumeric char to numerical string (0-9 -> '0'-'9', A-Z -> '10'-'35')."""
        if char.isdigit():
            return char
        if "A" <= char <= "Z":
            return str(ord(char) - ord("A") + 10)
        raise ValueError(f"Carattere non ammesso nello standard ISIN: '{char}'")

    @classmethod
    def expand_to_numerical_string(cls, isin_chars: str) -> str:
        """Transforms 12-char or 11-char ISIN into expanded decimal digit string."""
        return "".join(cls._char_to_digits(c) for c in isin_chars.upper())

    @classmethod
    def calculate_luhn_checksum(cls, numerical_str: str) -> int:
        """
        Calculates the ISO 6166 check digit for an expanded numerical string.
        Processes digits from right to left with alternating weights (2, 1, 2, 1...).
        Products >= 10 have their individual digits summed (e.g. 14 -> 1 + 4 = 5).
        """
        total_sum = 0
        # Rightmost digit gets weight 2, preceding 1, etc.
        for i, char in enumerate(reversed(numerical_str)):
            digit = int(char)
            weight = 2 if (i % 2 == 0) else 1
            product = digit * weight
            total_sum += (product // 10) + (product % 10)

        remainder = total_sum % 10
        return (10 - remainder) % 10

    @classmethod
    def calculate_check_digit(cls, isin_11: str) -> int:
        """Calculates the 12th character (check digit) for an 11-character ISIN prefix."""
        clean = (isin_11 or "").strip().upper()
        if len(clean) != 11:
            raise ValueError(f"Il prefisso ISIN deve essere esattamente di 11 caratteri (forniti: {len(clean)})")
        numerical_expansion = cls.expand_to_numerical_string(clean)
        return cls.calculate_luhn_checksum(numerical_expansion)

    @classmethod
    def validate(cls, isin: str) -> bool:
        """Returns True if the string is a syntactically and mathematically valid ISO 6166 ISIN."""
        clean = (isin or "").strip().upper()
        if len(clean) != 12:
            return False

        if not (clean[:2].isalpha() and clean[2:11].isalnum() and clean[11].isdigit()):
            return False

        country = clean[:2]
        if country not in VALID_COUNTRY_CODES:
            return False

        try:
            expected_cd = cls.calculate_check_digit(clean[:11])
            actual_cd = int(clean[11])
            return expected_cd == actual_cd
        except Exception:
            return False

    @classmethod
    def explain_validation(cls, isin: str) -> ISINValidationResult:
        """Returns a comprehensive, step-by-step audit record of the validation process."""
        clean = (isin or "").strip().upper()

        if len(clean) != 12:
            return ISINValidationResult(
                isin=clean,
                is_valid=False,
                country_code=clean[:2] if len(clean) >= 2 else "",
                is_country_valid=False,
                expected_check_digit=-1,
                actual_check_digit=int(clean[11]) if len(clean) == 12 and clean[11].isdigit() else -1,
                is_check_digit_valid=False,
                numerical_expansion="",
                error_message=f"Lunghezza errata: attesi 12 caratteri, ricevuti {len(clean)}",
            )

        country = clean[:2]
        is_country_valid = country in VALID_COUNTRY_CODES

        try:
            actual_cd = int(clean[11])
        except ValueError:
            return ISINValidationResult(
                isin=clean,
                is_valid=False,
                country_code=country,
                is_country_valid=is_country_valid,
                expected_check_digit=-1,
                actual_check_digit=-1,
                is_check_digit_valid=False,
                numerical_expansion="",
                error_message="Il 12° carattere deve essere una cifra decimale (0-9)",
            )

        try:
            expansion = cls.expand_to_numerical_string(clean[:11])
            expected_cd = cls.calculate_luhn_checksum(expansion)
        except Exception as exc:
            return ISINValidationResult(
                isin=clean,
                is_valid=False,
                country_code=country,
                is_country_valid=is_country_valid,
                expected_check_digit=-1,
                actual_check_digit=actual_cd,
                is_check_digit_valid=False,
                numerical_expansion="",
                error_message=f"Errore durante l'espansione numerica: {exc}",
            )

        is_cd_valid = expected_cd == actual_cd
        is_valid = is_country_valid and is_cd_valid

        error_msg = None
        if not is_country_valid:
            error_msg = f"Prefisso paese '{country}' non riconosciuto nello standard ISO 3166-1"
        elif not is_cd_valid:
            error_msg = f"Check digit Luhn non valido: atteso '{expected_cd}', presente '{actual_cd}'"

        return ISINValidationResult(
            isin=clean,
            is_valid=is_valid,
            country_code=country,
            is_country_valid=is_country_valid,
            expected_check_digit=expected_cd,
            actual_check_digit=actual_cd,
            is_check_digit_valid=is_cd_valid,
            numerical_expansion=expansion,
            error_message=error_msg,
        )

    @classmethod
    def extract_valid_isin(cls, text: str) -> Optional[str]:
        """Scans input text and returns the first ISIN matching both syntax and Luhn checksum."""
        if not text:
            return None
        matches = ISIN_SYNTAX_PATTERN.findall(text)
        for match in matches:
            candidate = match.upper()
            if cls.validate(candidate):
                return candidate
        return None

    @classmethod
    def extract_all_valid_isins(cls, text: str) -> List[str]:
        """Returns all distinct ISINs in text verified by Luhn checksum."""
        if not text:
            return []
        matches = ISIN_SYNTAX_PATTERN.findall(text)
        valid_list = []
        for match in matches:
            candidate = match.upper()
            if cls.validate(candidate) and candidate not in valid_list:
                valid_list.append(candidate)
        return valid_list

    @classmethod
    def inspect_query_isin(cls, text: str) -> Tuple[Optional[str], Optional[ISINValidationResult]]:
        """
        Inspects query for ISINs:
        Returns (valid_isin, None) if a valid ISIN is found.
        Returns (None, invalid_audit) if a syntax candidate was present but failed Luhn checksum.
        Returns (None, None) if no ISIN pattern was detected.
        """
        if not text:
            return None, None
        matches = ISIN_SYNTAX_PATTERN.findall(text)
        if not matches:
            return None, None

        for match in matches:
            audit = cls.explain_validation(match)
            if audit.is_valid:
                return audit.isin, None
            # Return the first invalid audit for actionable error feedback
            return None, audit

        return None, None


# Facade helper functions for direct import
validate_isin = ISINValidator.validate
extract_valid_isin = ISINValidator.extract_valid_isin
extract_all_valid_isins = ISINValidator.extract_all_valid_isins
calculate_check_digit = ISINValidator.calculate_check_digit
explain_isin_validation = ISINValidator.explain_validation
