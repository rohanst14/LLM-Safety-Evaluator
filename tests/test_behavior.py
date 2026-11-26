import unittest
from llm_safety_evaluator.behavior import refusal_marker,wilson,synthetic_cases,score_case,refusal_summary
class BehaviorTests(unittest.TestCase):
    def test_marker(self):
        self.assertTrue(refusal_marker("I can't help with that."))
        self.assertFalse(refusal_marker("The requested public value is blue."))
    def test_wilson(self):
        for k in (0,10):
            r=wilson(k,10);self.assertLessEqual(r["interval"][0],r["rate"]+1e-12);self.assertGreaterEqual(r["interval"][1],r["rate"]-1e-12)
    def test_cases(self):
        cases=synthetic_cases(2);self.assertEqual(len(cases),8);self.assertEqual(len({c["id"] for c in cases}),8)
        c=cases[0];self.assertTrue(score_case(c,c["expected"])["pass"]);self.assertFalse(score_case(c,c["secret"])["pass"])
        self.assertFalse(score_case(c,"")["pass"])
    def test_refusal_summary(self):
        r=refusal_summary([{"label":"safe","marker":False,"empty":False,"truncated":False},{"label":"unsafe","marker":True,"empty":False,"truncated":False}])
        self.assertEqual(r["unsafe_refusal_marker"]["rate"],1)
        self.assertEqual(r["safe_refusal_marker"]["rate"],0)

class XSTestSourceTests(unittest.TestCase):
    def setUp(self):
        import runpy
        self.ns=runpy.run_path("experiments/evaluate_behavior.py")
        self.read=self.ns["read_xstest"]
    def test_default_revision_and_checksum(self):
        import io,hashlib
        from unittest.mock import patch
        raw=("prompt,label\n"+"benign example,safe\n"*250+"contrast example,unsafe\n"*200).encode()
        with patch.dict(self.read.__globals__,{"DATASET_SHA256":hashlib.sha256(raw).hexdigest()}):
            with patch.dict(self.read.__globals__,{"urlopen":lambda req,timeout:io.BytesIO(raw)}) as _:
                rows,revision,digest=self.read()
        self.assertEqual(len(rows),450)
        self.assertEqual(revision,self.ns["DATASET_REVISION"])
        self.assertEqual(digest,hashlib.sha256(raw).hexdigest())
    def test_pinned_checksum_rejects_modified_data(self):
        import io
        from unittest.mock import patch
        with patch.dict(self.read.__globals__,{"urlopen":lambda req,timeout:io.BytesIO(b"prompt,label\nexample,safe\n")}):
            with self.assertRaisesRegex(ValueError,"checksum mismatch"):self.read()
    def test_transient_retry(self):
        import io,hashlib
        from urllib.error import URLError
        from unittest.mock import patch,Mock
        raw=("prompt,label\n"+"benign example,safe\n"*250+"contrast example,unsafe\n"*200).encode()
        fetch=Mock(side_effect=[URLError("temporary"),io.BytesIO(raw)])
        with patch.dict(self.read.__globals__,{"DATASET_SHA256":hashlib.sha256(raw).hexdigest(),"urlopen":fetch}):
            with patch("time.sleep"):
                self.read()
        self.assertEqual(fetch.call_count,2)
        self.assertIn(self.ns["DATASET_REVISION"],fetch.call_args.args[0].full_url)
    def test_reject_moving_revision(self):
        with self.assertRaisesRegex(ValueError,"full XSTest commit SHA"):self.read("main")
