"""
Unit tests for the ISO 6166 ISIN validation and checksum module.
Compatible with pytest and python -m unittest.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.utils.isin import (
    ISINValidator,
    calculate_check_digit,
    explain_isin_validation,
    extract_all_valid_isins,
    extract_valid_isin,
    validate_isin,
)


class TestISINValidation(unittest.TestCase):
    # Real-world benchmark ISINs from various global jurisdictions
    VALID_ISINS = [
        ("IE00B4L5Y983", "iShares Core MSCI World (Ireland)"),
        ("IE00B5BMR087", "iShares Core S&P 500 (Ireland)"),
        ("LU1681043599", "Amundi MSCI World (Luxembourg)"),
        ("US0378331005", "Apple Inc (United States)"),
        ("US4642872000", "iShares MSCI Emerging Markets (United States)"),
        ("FR0010315770", "Lyxor CAC 40 (France)"),
        ("DE000A0F5UF5", "iShares NASDAQ-100 (Germany)"),
        ("IT0005239360", "BTP Italia (Italy)"),
        ("XS2243564478", "Euroclear/Clearstream Bond (International)"),
    ]

    def test_valid_isins(self):
        for isin, desc in self.VALID_ISINS:
            with self.subTest(isin=isin, desc=desc):
                self.assertTrue(validate_isin(isin), f"Expected valid ISIN for {desc} ({isin})")
                audit = explain_isin_validation(isin)
                self.assertTrue(audit.is_valid)
                self.assertTrue(audit.is_check_digit_valid)
                self.assertEqual(audit.expected_check_digit, audit.actual_check_digit)

    def test_invalid_check_digits(self):
        # Tampered check digits
        tampered_cases = [
            ("IE00B4L5Y984", 3, 4),  # expected 3, got 4
            ("IE00B5BMR080", 7, 0),  # expected 7, got 0
            ("LU1681043590", 9, 0),  # expected 9, got 0
            ("US0378331009", 5, 9),  # expected 5, got 9
        ]
        for isin, expected, actual in tampered_cases:
            with self.subTest(isin=isin):
                self.assertFalse(validate_isin(isin))
                audit = explain_isin_validation(isin)
                self.assertFalse(audit.is_valid)
                self.assertFalse(audit.is_check_digit_valid)
                self.assertEqual(audit.expected_check_digit, expected)
                self.assertEqual(audit.actual_check_digit, actual)

    def test_invalid_syntax_and_lengths(self):
        invalid_syntaxes = [
            "",
            "   ",
            "IE00B4L5Y98",      # 11 chars (too short)
            "IE00B4L5Y9833",    # 13 chars (too long)
            "123456789012",     # No country code prefix
            "XX0000000000",     # Invalid country code XX
            "IE00B4L5Y98A",     # Last char not a digit
            "IE--B4L5Y983",     # Special characters
        ]
        for isin in invalid_syntaxes:
            with self.subTest(isin=isin):
                self.assertFalse(validate_isin(isin))

    def test_calculate_check_digit(self):
        self.assertEqual(calculate_check_digit("IE00B4L5Y98"), 3)
        self.assertEqual(calculate_check_digit("IE00B5BMR08"), 7)
        self.assertEqual(calculate_check_digit("LU168104359"), 9)
        self.assertEqual(calculate_check_digit("US037833100"), 5)
        self.assertEqual(calculate_check_digit("DE000A0F5UF"), 5)

        with self.assertRaises(ValueError):
            calculate_check_digit("SHORT")

    def test_extract_valid_isin_from_text(self):
        text = "Vorrei analizzare l'ETF con codice IE00B4L5Y983 quotato a Milano."
        extracted = extract_valid_isin(text)
        self.assertEqual(extracted, "IE00B4L5Y983")

        # Lowercase extraction normalized to uppercase
        text_lower = "analizza ie00b5bmr087 per favore"
        self.assertEqual(extract_valid_isin(text_lower), "IE00B5BMR087")

        # Text with fake ISIN should not be extracted
        fake_text = "Ecco un codice finto IE00B4L5Y989 nel messaggio"
        self.assertIsNone(extract_valid_isin(fake_text))

    def test_extract_all_valid_isins(self):
        multi_text = "Confronta IE00B4L5Y983 con LU1681043599 e il finto XX0000000000."
        isins = extract_all_valid_isins(multi_text)
        self.assertEqual(isins, ["IE00B4L5Y983", "LU1681043599"])

    def test_inspect_query_isin(self):
        # Valid query
        valid_isin, audit = ISINValidator.inspect_query_isin("Analizza IE00B4L5Y983")
        self.assertEqual(valid_isin, "IE00B4L5Y983")
        self.assertIsNone(audit)

        # Invalid candidate query (tampered check digit)
        valid_isin, audit = ISINValidator.inspect_query_isin("Analizza IE00B4L5Y984")
        self.assertIsNone(valid_isin)
        self.assertIsNotNone(audit)
        self.assertEqual(audit.isin, "IE00B4L5Y984")
        self.assertFalse(audit.is_valid)

        # Generic chit-chat query
        valid_isin, audit = ISINValidator.inspect_query_isin("Ciao buongiorno!")
        self.assertIsNone(valid_isin)
        self.assertIsNone(audit)


if __name__ == "__main__":
    unittest.main()
