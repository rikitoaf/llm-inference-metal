import unittest
from decimal import Decimal

from inventory.report import total_value
from inventory.store import Store


class TestInventory(unittest.TestCase):
    def test_add_and_remove(self):
        s = Store()
        s.add("A1", "Bolt", 10, "0.25")
        s.remove("A1", 4)
        self.assertEqual(s.items()[0][2], 6)

    def test_total(self):
        s = Store()
        s.add("A1", "Bolt", 4, "0.25")
        self.assertEqual(total_value(s), Decimal("1.00"))


if __name__ == "__main__":
    unittest.main()
