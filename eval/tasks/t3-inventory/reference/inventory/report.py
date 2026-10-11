import csv
import io
from decimal import ROUND_HALF_UP, Decimal


def total_value(store):
    total = sum((qty * price for _, _, qty, price in store.items()), Decimal(0))
    return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def to_csv(store):
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["sku", "name", "qty", "price"])
    for sku, name, qty, price in store.items():
        w.writerow([sku, name, qty, f"{price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}"])
    return out.getvalue()
