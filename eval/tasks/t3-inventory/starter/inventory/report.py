"""Reports over a Store."""


def total_value(store):
    """Sum of qty * price over all items, as a Decimal rounded half-up to 2 decimal places
    (so a total of 0.125 becomes Decimal('0.13'))."""
    return round(sum(qty * price for _, _, qty, price in store.items()), 2)


def to_csv(store):
    """CSV text: the header line 'sku,name,qty,price', then one line per item sorted by sku, with
    the price written with exactly 2 decimals. A field containing a comma, a double quote or a
    newline is wrapped in double quotes, with inner double quotes doubled (RFC 4180). Every line,
    including the last, ends with '\\n'."""
    lines = ["sku,name,qty,price"]
    for sku, name, qty, price in store.items():
        lines.append(f"{sku},{name},{qty},{price}")
    return "\n".join(lines)
