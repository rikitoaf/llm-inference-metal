from decimal import Decimal


class OutOfStock(Exception):
    pass


class Store:
    def __init__(self):
        self._items = {}

    def add(self, sku, name, qty, price):
        if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
            raise ValueError(qty)
        price = Decimal(price)
        if sku in self._items:
            self._items[sku]["qty"] += qty
            self._items[sku]["price"] = price
        else:
            self._items[sku] = {"name": name, "qty": qty, "price": price}

    def remove(self, sku, qty):
        item = self._items[sku]
        if qty > item["qty"]:
            raise OutOfStock(sku)
        item["qty"] -= qty
        if item["qty"] == 0:
            del self._items[sku]

    def items(self):
        return [(s, i["name"], i["qty"], i["price"]) for s, i in sorted(self._items.items())]
