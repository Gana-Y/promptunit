import os
import subprocess
import sys

from promptunit.cli import main


def test_plugin_summary_and_replay_flag(pytester):
    pytester.makepyfile(
        """
        from promptunit import MockLLM, expect

        def test_ok():
            expect(MockLLM("hello world")("hi")).contains("hello")
        """
    )
    result = pytester.runpytest("--promptunit-mode=replay")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*promptunit*", "*mode=replay*calls=1*"])


def test_plugin_failure_message(pytester):
    pytester.makepyfile(
        """
        from promptunit import MockLLM, expect

        def test_bad():
            expect(MockLLM("Sorry, no.")("hi")).contains("refund")
        """
    )
    result = pytester.runpytest()
    result.assert_outcomes(failed=1)
    out = result.stdout.str()
    assert "expected output to contain ['refund']" in out
    assert "Sorry, no." in out


def test_cli_init_and_cache(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["init", "mytests"]) == 0
    assert (tmp_path / "mytests" / "test_prompts.py").exists()
    assert main(["init", "mytests"]) == 1  # refuses to overwrite
    assert main(["cache", "info", "--dir", str(tmp_path / "c")]) == 0
    assert "0 cached responses" in capsys.readouterr().out


def test_generated_template_runs_offline(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "PROMPTUNIT_OLLAMA_MODEL")}
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing_pp}" if existing_pp else src_dir
    subprocess.run([sys.executable, "-m", "promptunit.cli", "init", str(tmp_path / "t")], check=True, env=env, cwd=tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path / "t"), "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, env=env, cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "3 passed" in proc.stdout
