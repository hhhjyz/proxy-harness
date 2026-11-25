"""Critical acceptance, subprocess isolation and evaluator boundary tests."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from proxy_harness.engine import run_agent
from proxy_harness.evaluate import evaluate, validate_eval
from proxy_harness.orchestrator import candidate_rank
from proxy_harness.spec import load_spec
from proxy_harness.util import ROOT, run_process


class WorkflowTests(unittest.TestCase):
    def test_missing_or_failed_evaluator_never_enters_best_pool(self):
        for status in ("skipped","error"):
            self.assertFalse(candidate_rank({"hard_pass":True},{"status":status,"subject_complete":True,"effective_score":100})[0])

    def test_high_score_cannot_override_geometry_or_missing_subject(self):
        good = {"status":"evaluated","subject_complete":True,"effective_score":99}
        self.assertFalse(candidate_rank({"hard_pass":False},good)[0])
        self.assertFalse(candidate_rank({"hard_pass":True},{**good,"subject_complete":False})[0])
        # A degraded last candidate must never replace the best valid one.
        best = candidate_rank({"hard_pass":True},{**good,"effective_score":80})
        self.assertGreater(best,candidate_rank({"hard_pass":False},good))

    def test_eval_rejects_nonfinite_and_boolean_score(self):
        for score in (float("nan"),float("inf"),True,-1,101):
            with self.assertRaises(ValueError):
                validate_eval({"score":score,"subject_complete":True,"summary":"x","issues":[]})

    def test_unconfigured_evaluator_is_explicitly_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            spec=load_spec(ROOT / "examples/demo.spec.json")
            result=evaluate(spec,{"hard_pass":True},out/"missing.png",out/"missing.png",out,{"backend":"none"})
            self.assertEqual(result["status"],"skipped")
            self.assertIsNone(result["effective_score"])
            self.assertEqual(json.loads((out/"eval_request.json").read_text())["history"],[])

    def test_command_argv_is_not_a_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            marker=path/"unwanted"
            payload=f"$(touch {marker})"
            r=run_process([sys.executable,"-c","import sys; print(sys.argv[1])",payload],cwd=path)
            self.assertEqual(r["stdout"].strip(),payload)
            self.assertFalse(marker.exists())

    def test_command_adapter_continuation_and_raw_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            work, trace=root/"work",root/"trace"
            config={"backend":"command","argv":[sys.executable,"-c","import json; print(json.dumps({'type':'result','session_id':'s1','text':'ok'}))"],
                    "resume_argv":[sys.executable,"-c","import sys,json; print(json.dumps({'type':'result','session_id':sys.argv[1],'text':'repaired'}))","{session_id}"]}
            one=run_agent(config,"first",work,trace)
            two=run_agent(config,"repair",work,root/"trace2",session_id=one["session_id"])
            self.assertEqual(two["session_id"],"s1")
            self.assertTrue(two["session_continued"])
            self.assertEqual(two["output"],"repaired")
            self.assertTrue((trace/"coder.stdout.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
