def formatted_string_to_int(s):
    
    if isinstance(s, int):
        return s
    
    assert isinstance(s, str), f"Argument 's' must be of type str or int, got {s}"
    
    s = s.strip().lower()
    
    multipliers = {
        'k': 1_000,
        'm': 1_000_000,
        'b': 1_000_000_000,
        't': 1_000_000_000_000
    }
    
    if s[-1] in multipliers:
        num = float(s[:-1])
        return int(num * multipliers[s[-1]])
    else:
        return int(float(s))

def int_to_formatted_string(n):

    if isinstance(n, str):
        return n

    assert isinstance(n, int), f"Argument 'n' must be of type int, got {type(n)}"

    suffixes = [
        (1_000_000_000_000, "t"),
        (1_000_000_000, "b"),
        (1_000_000, "m"),
        (1_000, "k"),
    ]

    for factor, suffix in suffixes:
        if abs(n) >= factor:
            value = n / factor
            return f"{value:.3g}{suffix}"
    
    return str(n)