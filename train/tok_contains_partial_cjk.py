from transformers import AutoTokenizer

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

class UTF8Splitter:
    """Splits byte sequences into logical UTF-8 units with proper partial handling.
    
    Rules:
    - Only the first token may be a right partial (continuation bytes without start)
    - Only the last token may be a left partial (incomplete multi-byte sequence)
    - All middle tokens must be complete UTF-8 sequences
    - Invalid sequences in middle raise ValueError
    """
    
    # UTF-8 state machine states
    STATE_START = 0          # Ready for new codepoint
    STATE_RIGHT_PARTIAL = 1  # Collecting continuation bytes (only valid at start)
    STATE_LEFT_PARTIAL = 2   # Incomplete multi-byte sequence (only valid at end)
    
    def __init__(self):
        self.tokens = []
        self.current_token = bytearray()
        self.state = self.STATE_START
        self.expected_continuations = 0
    
    def feed(self, byte: int):
        """Process a single byte through the state machine."""
        # Handle continuation byte (10xxxxxx)
        if 0x80 <= byte < 0xC0:
            if self.state == self.STATE_START:
                # First byte is continuation → right partial
                self.state = self.STATE_RIGHT_PARTIAL
                self.current_token.append(byte)
            elif self.state == self.STATE_RIGHT_PARTIAL:
                # Continuing right partial
                self.current_token.append(byte)
            elif self.expected_continuations > 0:
                # Valid continuation in multi-byte sequence
                self.current_token.append(byte)
                self.expected_continuations -= 1
                if self.expected_continuations == 0:
                    # Completed a multi-byte sequence
                    self._finalize_token()
            else:
                # Unexpected continuation byte in middle
                raise ValueError(
                    f"Unexpected continuation byte (0x{byte:02x}) at position {len(self.current_token)}"
                )
        
        # Handle ASCII (0xxxxxxx)
        elif byte < 0x80:
            if self.current_token:
                self._finalize_token()
            self.current_token.append(byte)
            self._finalize_token()
        
        # Handle multi-byte start (11xxxxxx)
        else:
            # Finalize any current token first
            if self.current_token:
                self._finalize_token()
            
            # Determine expected continuation bytes
            if 0xC0 <= byte < 0xE0:    # 2-byte sequence
                self.expected_continuations = 1
            elif 0xE0 <= byte < 0xF0:  # 3-byte sequence
                self.expected_continuations = 2
            elif 0xF0 <= byte < 0xF8:  # 4-byte sequence
                self.expected_continuations = 3
            else:  # Invalid start byte (0xF8+)
                raise ValueError(
                    f"Invalid UTF-8 start byte (0x{byte:02x}) at position {len(self.current_token)}"
                )
            
            # Start new token
            self.current_token.append(byte)
            self.state = self.STATE_START
    
    def _finalize_token(self):
        """Finalize the current token and reset state."""
        if self.current_token:
            self.tokens.append(bytearray(self.current_token))
            self.current_token = bytearray()
            self.expected_continuations = 0
    
    def flush(self):
        """Finalize processing and return all tokens with completeness flags."""
        if self.current_token:
            # Left partial is only allowed at the very end
            self.tokens.append(bytearray(self.current_token))
            self.current_token = bytearray()
        
        # Verify only first/last tokens can be partial
        if len(self.tokens) > 1:
            # Middle tokens must be complete
            for i in range(1, len(self.tokens) - 1):
                self._validate_complete(self.tokens[i])
        
        # === MINIMAL CHANGE STARTS HERE ===
        # Return list of (token, type) pairs with three possible types
        return [(token, self._get_token_type(token)) for token in self.tokens]
        # === MINIMAL CHANGE ENDS HERE ===
    
    def _validate_complete(self, token: bytearray):
        """Ensure a token is a complete UTF-8 sequence."""
        if not token:
            return
        
        first_byte = token[0]
        
        # ASCII is always complete
        if first_byte < 0x80:
            return
        
        # Continuation byte as first byte → invalid in middle
        if 0x80 <= first_byte < 0xC0:
            raise ValueError(f"Unexpected continuation byte in middle token: {token}")
        
        # Determine expected length from start byte
        if 0xC0 <= first_byte < 0xE0:  # 2-byte
            expected_len = 2
        elif 0xE0 <= first_byte < 0xF0:  # 3-byte
            expected_len = 3
        elif 0xF0 <= first_byte < 0xF8:  # 4-byte
            expected_len = 4
        else:
            raise ValueError(f"Invalid start byte in middle token: {token}")
        
        # Check actual length
        if len(token) != expected_len:
            raise ValueError(
                f"Incomplete UTF-8 sequence in middle token: {token} "
                f"(expected {expected_len} bytes, got {len(token)})"
            )
        
        # Verify continuation bytes
        for i in range(1, len(token)):
            if not (0x80 <= token[i] < 0xC0):
                raise ValueError(
                    f"Invalid continuation byte in middle token: {token} "
                    f"(byte {i} is 0x{token[i]:02x})"
                )
    
    # === MINIMAL CHANGE: REPLACE _is_complete WITH THIS ===
    def _get_token_type(self, token: bytearray) -> str:
        """Determine the type of UTF-8 sequence represented by the token."""
        if not token:
            return "complete"  # Empty token is degenerate but considered complete
            
        first = token[0]
        
        # ASCII is always complete
        if first < 0x80:
            return "complete"
            
        # Continuation byte as first byte → right partial
        if 0x80 <= first < 0xC0:
            return "right_partial"
            
        # Determine expected length from start byte
        if first < 0xE0:    # 2-byte sequence
            expected = 2
        elif first < 0xF0:  # 3-byte sequence
            expected = 3
        elif first < 0xF8:  # 4-byte sequence
            expected = 4
        else:               # Invalid start byte (shouldn't happen)
            return "left_partial"  # Treat as incomplete
            
        return "complete" if len(token) == expected else "left_partial"


def process_bytes(byte_seq: bytearray) -> list[tuple[bytearray, str]]:
    """Split byte sequence into logical UTF-8 units with type classification.
    
    Returns list of (token, type) pairs where type is one of:
    - "complete": complete UTF-8 sequence
    - "right_partial": continuation bytes without start (only allowed at beginning)
    - "left_partial": incomplete multi-byte sequence (only allowed at end)
    """
    splitter = UTF8Splitter()
    
    for byte in byte_seq:
        splitter.feed(byte)
    
    return splitter.flush()

def utf8_code_point_bounds(b1: int, b2: int) -> tuple[int | None, int | None]:
    """
    Given the first two bytes of a UTF-8 sequence, return the lowest and highest
    possible Unicode code points that could result from a valid UTF-8 character
    starting with these two bytes.

    Returns:
        (lower_bound, upper_bound) if valid
        (None, None) if invalid or incomplete in a way that can't form a character
    """
    # Validate b1 and b2 are in byte range
    if not (0 <= b1 <= 255 and 0 <= b2 <= 255):
        return (None, None)

    # Second byte must be a continuation byte: 10xxxxxx
    if (b2 & 0xC0) != 0x80:
        return (None, None)  # Invalid second byte

    # 2-byte sequence: 110xxxxx 10xxxxxx
    if 0xC2 <= b1 <= 0xDF:
        c = (b1 & 0x1F)  # 5 bits
        d = (b2 & 0x3F)  # 6 bits
        lo = (c << 6) | d
        hi = lo  # Only one possible code point from these two bytes
        return (lo, hi)

    # 3-byte sequence: 1110xxxx 10xxxxxx 10xxxxxx
    elif 0xE0 <= b1 <= 0xEF:
        lead = b1 & 0x0F  # 4 bits
        mid  = b2 & 0x3F   # 6 bits
        # Third byte can be 0x80 to 0xBF → 6 bits: 0b00xxxxxx
        lo = (lead << 12) | (mid << 6) | 0x00  # last 6 bits = 0
        hi = (lead << 12) | (mid << 6) | 0x3F  # last 6 bits = 1
        # But: exclude overlong and out-of-range
        if b1 == 0xE0:
            lo = max(lo, 0x0800)  # No overlongs < U+0800
        if b1 == 0xED and b2 >= 0xA0:
            hi = min(hi, 0xD7FF)  # Surrogates (U+D800–U+DFFF) are invalid
        if lo > 0xFFFF:
            return (None, None)  # Out of 3-byte range
        return (lo, hi)

    # 4-byte sequence: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
    elif 0xF0 <= b1 <= 0xF4:
        lead = b1 & 0x07    # 3 bits
        mid  = b2 & 0x3F    # 6 bits
        # Last two bytes: each 6 bits → total 12 bits free
        lo = (lead << 18) | (mid << 12) | 0x000  # last 12 bits = 0
        hi = (lead << 18) | (mid << 12) | 0xFFF  # last 12 bits = 1

        # Apply constraints:
        # - Minimum for 4-byte is U+10000
        # - Maximum Unicode is U+10FFFF
        if b1 == 0xF0:
            lo = max(lo, 0x10000)  # No overlongs
        elif b1 == 0xF4:
            hi = min(hi, 0x10FFFF)  # Can't go beyond Unicode
        elif b1 > 0xF4:
            return (None, None)  # Invalid leading byte

        if hi < 0x10000 or lo > 0x10FFFF:
            return (None, None)

        return (lo, hi)

    else:
        return (None, None)  # Invalid first byte

def infer_utf8_range(b1: int, b2: int, ranges: dict[str, tuple[int, int]]) -> list[str]:
    """
    Given the first two bytes of a UTF-8 sequence, return a list of range names
    (e.g., 'cjk_main', 'hiragana') that the full character *could* belong to.

    Args:
        b1: First byte
        b2: Second byte
        ranges: Dict of {name: (start, end)} for Unicode blocks

    Returns:
        List of matching range names (possibly empty)
    """
    lo, hi = utf8_code_point_bounds(b1, b2)
    if lo is None or hi is None:
        return []

    matches = []
    for name, (start, end) in ranges.items():
        # Check for overlap: not (hi < start or lo > end)
        if not (hi < start or lo > end):
            matches.append(name)
    return matches

# Define your RANGES (from earlier)
RANGES = {
    'cjk_main': (0x4E00, 0x9FFF),
    'cjk_ext_b': (0x20000, 0x2A6DF),
    'cjk_ext_c': (0x2A700, 0x2B73F),
    'cjk_ext_d': (0x2B740, 0x2B81F),
    'cjk_ext_e': (0x2B820, 0x2CEAF),
    'cjk_ext_f': (0x2CEB0, 0x2EBEF),
    'cjk_ext_i': (0x2EBF0, 0x2EE5F),
    'cjk_compat_supp': (0x2F800, 0x2FA1F),
    'cjk_ext_g': (0x30000, 0x3134F),
    'cjk_ext_h': (0x31350, 0x323AF),

    'hiragana': (0x3040, 0x309F),
    'katakana': (0x30A0, 0x30FF),
    'katakana_phonetic': (0x31F0, 0x31FF),
    'kana_supplement': (0x1B000, 0x1B0FF),
    'kana_ext_a': (0x1B100, 0x1B12F),
    'small_kana_ext': (0x1B130, 0x1B16F),
    'kana_ext_b': (0x1AFF0, 0x1AFFF),

    'emoji_emoticons': (0x1F600, 0x1F64F),
    'emoji_symbols': (0x1F680, 0x1F6FF),
    'misc_symbols': (0x1F300, 0x1F5FF),
}
    
byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

def tok_to_bytes(tokenizer, tok):
    text = tokenizer._convert_id_to_token(tok)
    return bytearray([byte_decoder[c] for c in text])

def print_possible_unicode(first_byte, second_byte):
    """
    Given the first two bytes of a UTF-8 sequence, prints all possible Unicode code points
    that could form a 3-byte or 4-byte UTF-8 sequence starting with these bytes.
    
    Args:
        first_byte: First byte of the UTF-8 sequence (0-255)
        second_byte: Second byte of the UTF-8 sequence (0-255)
    """
    results_3byte = []
    results_4byte = []
    
    # Check if these could be the first two bytes of a 3-byte UTF-8 sequence
    if 0xE0 <= first_byte <= 0xEF:
        # Validate second byte for 3-byte sequence
        if 0x80 <= second_byte <= 0xBF:
            # Special validity checks for canonical UTF-8
            if first_byte == 0xE0 and second_byte < 0xA0:
                pass  # Will be filtered later
            elif first_byte == 0xED and second_byte > 0x9F:
                pass  # Will be filtered later
            else:
                # Extract the relevant bits
                xxxx = first_byte & 0x0F      # 4 bits from first byte (after 1110)
                yyyyyy = second_byte & 0x3F   # 6 bits from second byte (after 10)
                
                # For each possible third byte (10zzzzzz)
                for zzzzzz in range(0x40):  # 0 to 63
                    # Calculate the full Unicode code point
                    # Format: xxxx yyyy yyzz zzzz
                    code_point = (xxxx << 12) | (yyyyyy << 6) | zzzzzz
                    
                    # Skip invalid Unicode ranges
                    if 0xD800 <= code_point <= 0xDFFF:  # Surrogates are not valid characters
                        continue
                    if code_point > 0x10FFFF:  # Beyond the Unicode range
                        continue
                        
                    # Try to get the actual character
                    try:
                        char = chr(code_point)
                        results_3byte.append((code_point, char))
                    except ValueError:
                        continue
    
    # Check if these could be the first two bytes of a 4-byte UTF-8 sequence
    if 0xF0 <= first_byte <= 0xF4:
        # Validate second byte for 4-byte sequence
        if 0x80 <= second_byte <= 0xBF:
            # Special validity checks for canonical UTF-8
            if first_byte == 0xF0 and second_byte < 0x90:
                pass  # Will be filtered later
            elif first_byte == 0xF4 and second_byte > 0x8F:
                pass  # Will be filtered later
            else:
                # Extract the relevant bits
                www = first_byte & 0x07      # 3 bits from first byte (after 11110)
                xxxxxx = second_byte & 0x3F  # 6 bits from second byte (after 10)
                
                # For each possible third and fourth bytes
                for yyyyyy in range(0x40):  # 0 to 63
                    for zzzzzz in range(0x40):  # 0 to 63
                        # Calculate the full Unicode code point
                        # Format: www xxxxxx yyyy yyzz zzzz
                        code_point = (www << 18) | (xxxxxx << 12) | (yyyyyy << 6) | zzzzzz
                        
                        # Check if within valid Unicode range
                        if 0x10000 <= code_point <= 0x10FFFF:
                            try:
                                char = chr(code_point)
                                results_4byte.append((code_point, char))
                            except ValueError:
                                continue
    
    # Print results
    if not results_3byte and not results_4byte:
        print("No valid Unicode code points for the given bytes")
        return
        
    if results_3byte:
        print(f"All possible Unicode code points for 3-byte sequence with first two bytes 0x{first_byte:02X} 0x{second_byte:02X}:")
        for code_point, char in results_3byte:
            print(char, end=', ')
        print(f"\nTotal 3-byte possibilities: {len(results_3byte)}\n")
    
    if results_4byte:
        print(f"All possible Unicode code points for 4-byte sequence with first two bytes 0x{first_byte:02X} 0x{second_byte:02X}:")
        for code_point, char in results_4byte:
            if char != '𬤼':
                print(char, end=', ')
                # print(f"U+{code_point:04X}: {char} (hex: 0x{code_point:04X})")
        print(f"\nTotal 4-byte possibilities: {len(results_4byte)}")


def contain_partial_cjk(tokenizer, tok):
    if tok < 256:
        return False
    decode_t = tokenizer.decode(tok)
    if '�' in decode_t:
        for p in process_bytes(tok_to_bytes(tokenizer, tok)):
            if p[1] == 'left_partial' and len(p[0]) > 1:
                # print(tok, infer_utf8_range(p[0][0], p[0][1], RANGES))
                for cat in infer_utf8_range(p[0][0], p[0][1], RANGES):
                    if 'cjk' in cat:
                        # print(tok, [tokenizer.decode(tok)], infer_utf8_range(p[0][0], p[0][1], RANGES))
                        # print_possible_unicode(p[0][0], p[0][1])
                        return True
    return False


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('/share/shmatikov/collin/language_confusion_paper/gate_weights/gate-llama3-8b-20k_95p_norm_2025-08-25-03:50:13_plugged')
    tok = 17885
    print(contain_partial_cjk(tokenizer, tok))