from contextlib import contextmanager
from pathlib import Path
import shutil
import unittest
import uuid

from ecom_search.index import ProductIndex, build_index
from ecom_search.query import plan_query


@contextmanager
def workspace():
    root = Path(__file__).resolve().parents[1]
    folder = root/"outputs"/"test-tmp"/uuid.uuid4().hex
    folder.mkdir(parents=True)
    try:
        yield folder
    finally:
        if not folder.resolve().is_relative_to(root):
            raise ValueError("test cleanup path escaped project")
        shutil.rmtree(folder)


class IndexTests(unittest.TestCase):
    def test_persistence_exclusion_cache_and_sql_safety(self):
        products = [dict(id="a",title="Nike blue running shoes",brand="Nike"), dict(id="b",title="Adidas red running shoes",brand="Adidas"),dict(id="c",title="Sugar free snack",brand="Cup")]
        with workspace() as folder:
            path = Path(folder)/"index.sqlite"
            build_index(products,path)
            with ProductIndex(path) as index:
                result = index.search("running shoes without Nike")
                self.assertEqual({p["id"] for p in result},{"b"})
                self.assertEqual([p["id"] for p in index.search("snack without sugar")],["c"])
                result[0]["title"] = "tampered"
                again = index.search("running shoes without Nike")
                self.assertNotEqual(again[0]["title"],"tampered")
                self.assertEqual(index.cache_hits,1)
                index.search('" OR 1=1; DROP TABLE products; --')
                self.assertEqual(len(index.catalog()),3)
                with self.assertRaises(ValueError): index.search("shoes",0)
            with self.assertRaises(FileExistsError): build_index(products,path)

    def test_duplicate_ids_rejected(self):
        with workspace() as folder:
            with self.assertRaises(ValueError): build_index([dict(id="a",title="a"),dict(id="a",title="b")],Path(folder)/"index.sqlite")

    def test_query_limits_and_budget_boundary(self):
        plan = plan_query('Find "running shoes" without Nike under $100')
        self.assertEqual(plan.excluded_terms,("nike",))
        self.assertIn("running shoes",plan.phrases)
        self.assertTrue(plan.warnings)
        self.assertNotIn("find",plan.positive_terms)
        self.assertEqual(plan_query("shoes -nike").excluded_terms,("nike",))
        self.assertEqual(plan_query("shoes no red").excluded_terms,("red",))
        with self.assertRaises(ValueError): plan_query("  ")


if __name__ == "__main__":
    unittest.main()
