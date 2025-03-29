import unittest
from src.data.loader import load_data

class TestDataLoader(unittest.TestCase):
    def test_load_data(self):
        data = load_data("data/creditcard.csv")
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)

if __name__ == "__main__":
    unittest.main()