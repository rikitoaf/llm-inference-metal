import unittest
from decimal import Decimal

from inventory.report import to_csv, total_value
from inventory.store import OutOfStock, Store


def store(*rows):
    s = Store()
    for r in rows:
        s.add(*r)
    return s


class Hidden(unittest.TestCase):
    def test_add_and_remove(self):
        s = store(("A1", "Bolt", 10, "0.25"))
        s.remove("A1", 4)
        self.assertEqual(s.items(), [("A1", "Bolt", 6, Decimal("0.25"))])

    def test_price_is_decimal(self):
        s = store(("A1", "Bolt", 1, "0.10"))
        self.assertIsInstance(s.items()[0][3], Decimal)
        self.assertEqual(s.items()[0][3], Decimal("0.10"))

    def test_readd_keeps_name_updates_price(self):
        s = store(("A1", "Bolt", 1, "0.10"), ("A1", "Other", 2, "0.20"))
        self.assertEqual(s.items(), [("A1", "Bolt", 3, Decimal("0.20"))])

    def test_bad_qty(self):
        s = Store()
        for q in [0, -1, 1.5, "2"]:
            with self.assertRaises(ValueError, msg=repr(q)):
                s.add("A1", "Bolt", q, "1.00")
        self.assertEqual(s.items(), [])

    def test_remove_unknown(self):
        with self.assertRaises(KeyError):
            Store().remove("NOPE", 1)

    def test_out_of_stock_unchanged(self):
        s = store(("A1", "Bolt", 3, "1.00"))
        with self.assertRaises(OutOfStock):
            s.remove("A1", 4)
        self.assertEqual(s.items()[0][2], 3)

    def test_zero_deletes(self):
        s = store(("A1", "Bolt", 3, "1.00"), ("B2", "Nut", 1, "1.00"))
        s.remove("A1", 3)
        self.assertEqual([i[0] for i in s.items()], ["B2"])

    def test_sorted(self):
        s = store(("C3", "c", 1, "1"), ("A1", "a", 1, "1"), ("B2", "b", 1, "1"))
        self.assertEqual([i[0] for i in s.items()], ["A1", "B2", "C3"])

    def test_total(self):
        self.assertEqual(total_value(store(("A1", "Bolt", 4, "0.25"))), Decimal("1.00"))

    def test_total_half_up(self):
        t = total_value(store(("A1", "x", 1, "0.125")))
        self.assertEqual(t, Decimal("0.13"))
        self.assertIsInstance(t, Decimal)
        self.assertEqual(str(total_value(store(("A1", "x", 3, "0.1")))), "0.30")

    def test_total_empty(self):
        self.assertEqual(total_value(Store()), Decimal("0.00"))

    def test_csv_plain(self):
        s = store(("B2", "Nut", 2, "0.5"), ("A1", "Bolt", 1, "1"))
        self.assertEqual(to_csv(s), "sku,name,qty,price\nA1,Bolt,1,1.00\nB2,Nut,2,0.50\n")

    def test_csv_quoting(self):
        s = store(("A1", 'Bolt, "M6"', 1, "2.5"))
        self.assertEqual(to_csv(s), 'sku,name,qty,price\nA1,"Bolt, ""M6""",1,2.50\n')

    def test_csv_empty(self):
        self.assertEqual(to_csv(Store()), "sku,name,qty,price\n")


if __name__ == "__main__":
    unittest.main()
