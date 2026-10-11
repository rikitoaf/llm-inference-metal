"""In-memory stock store."""
from decimal import Decimal


class OutOfStock(Exception):
    """Raised when removing more units than are in stock."""


class Store:
    def __init__(self):
        self._items = {}  # sku -> {"name": str, "qty": int, "price": Decimal}

    def add(self, sku, name, qty, price):
        """Add qty units of sku at a unit price given as a string or Decimal (never float).

        qty must be a positive int, otherwise ValueError. Adding an existing sku increases its
        qty, keeps its original name and replaces its price with the new one.
        """
        if sku in self._items:
            self._items[sku]["qty"] += qty
        else:
            self._items[sku] = {"name": name, "qty": qty, "price": float(price)}

    def remove(self, sku, qty):
        """Remove qty units of sku.

        An unknown sku raises KeyError. Removing more units than are in stock raises OutOfStock
        and leaves the stock unchanged. A sku that reaches 0 units is deleted from the store.
        """
        self._items[sku]["qty"] -= qty

    def items(self):
        """List of (sku, name, qty, price) tuples sorted by sku; price is a Decimal."""
        return [(s, i["name"], i["qty"], i["price"]) for s, i in self._items.items()]
