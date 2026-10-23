"""Optional authenticated encryption: Scrypt KDF + ChaCha20-Poly1305 per frame.

* key   = Scrypt(passphrase, salt (16 B, random per session, sent in OFFER), n=2^15, r=8, p=1, 32 B)
* nonce = session_id (u32 LE) ‖ dir (1 B) ‖ 00 00 00 ‖ seq (u32 LE)
* AAD   = the 14-byte link frame header (whose payload_len is the ciphertext length)

``seq`` is unique per direction within a session and the salt is fresh per
session, so (key, nonce) pairs never repeat.
"""

from __future__ import annotations

import os
import struct

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from ultrasonic_airdrop.config import SALT_LEN, SCRYPT_N, SCRYPT_P, SCRYPT_R
from ultrasonic_airdrop.errors import CryptoError

TAG_LEN = 16


def new_salt() -> bytes:
    """Random 16-byte salt."""
    return os.urandom(SALT_LEN)


def derive_key(passphrase: str, salt: bytes) -> bytes:
    """Derive the 32-byte session key from a passphrase (NFC-normalised UTF-8) and salt."""
    import unicodedata

    if len(salt) != SALT_LEN:
        raise ValueError("salt must be 16 bytes")
    if not passphrase:
        raise ValueError("passphrase must not be empty")
    kdf = Scrypt(salt=salt, length=32, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
    return kdf.derive(unicodedata.normalize("NFC", passphrase).encode("utf-8"))


def make_nonce(session_id: int, direction: int, seq: int) -> bytes:
    """Build the 12-byte nonce."""
    if direction not in (0, 1):
        raise ValueError("direction must be 0 or 1")
    return struct.pack("<IB3xI", session_id, direction, seq)


class FrameCipher:
    """Encrypts/decrypts link-frame payloads for one session."""

    def __init__(self, key: bytes, session_id: int) -> None:
        if len(key) != 32:
            raise ValueError("key must be 32 bytes")
        self._aead = ChaCha20Poly1305(key)
        self.session_id = session_id

    @classmethod
    def from_passphrase(cls, passphrase: str, salt: bytes, session_id: int) -> FrameCipher:
        """Derive the key and build a cipher."""
        return cls(derive_key(passphrase, salt), session_id)

    def seal(self, header: bytes, plaintext: bytes, direction: int, seq: int) -> bytes:
        """Encrypt ``plaintext``; ``header`` must already carry the ciphertext length."""
        return self._aead.encrypt(make_nonce(self.session_id, direction, seq), plaintext, header)

    def open(self, header: bytes, ciphertext: bytes, direction: int, seq: int) -> bytes:
        """Decrypt and authenticate.  Raises :class:`CryptoError` on failure."""
        try:
            return self._aead.decrypt(make_nonce(self.session_id, direction, seq), ciphertext, header)
        except InvalidTag as exc:
            raise CryptoError() from exc
