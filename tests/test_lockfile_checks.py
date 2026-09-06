"""SCRUM-1269: the one part of a pull request nobody reads.

A lockfile is on `SKIP`, which is right for the model — 40,000 generated lines
buy nothing and cost the budget every other file needs — but the consequence is
that a dependency change is reviewed by no one at all. caeli-marketing#243
changed an image and `package-lock.json` and the run reported "nothing
reviewable".

These are lookups, not judgements, for the same reason `route_without_test` is
one: the questions worth asking of a lockfile have definite answers, and a
model asked to skim a diff this size summarises it rather than checks it.
"""
from agentic_review import checks
from agentic_review import review as pr


NPM = '''@@ -10,7 +10,7 @@
     "node_modules/left-pad": {
-      "resolved": "https://registry.npmjs.org/left-pad/-/left-pad-1.3.0.tgz",
+      "resolved": "https://evil.example.com/left-pad/-/left-pad-1.3.0.tgz",
'''

REPUBLISH = '''@@ -30,6 +30,6 @@
     "node_modules/ok": {
       "version": "2.0.0",
-      "integrity": "sha512-CCC=="
+      "integrity": "sha512-DDD=="
'''

BUMP = '''@@ -50,6 +50,7 @@
     "node_modules/bumped": {
-      "version": "1.0.0",
-      "integrity": "sha512-EEE=="
+      "version": "1.1.0",
+      "integrity": "sha512-FFF=="
'''


class TestAnUnusualRegistryIsWorthOneLook:
    def test_a_foreign_host_is_named(self):
        out = checks.foreign_registries({"package-lock.json": NPM})
        assert len(out) == 1 and out[0]["severity"] == "medium"
        assert "evil.example.com" in out[0]["detail"]

    def test_the_usual_registries_say_nothing(self):
        quiet = NPM.replace("evil.example.com", "registry.npmjs.org")
        assert checks.foreign_registries({"package-lock.json": quiet}) == []

    def test_a_subdomain_of_a_default_is_still_default(self):
        """`registry.npmjs.org` and `cdn.registry.npmjs.org` are the same
        supplier; flagging one would be noise."""
        ok = NPM.replace("evil.example.com", "cdn.registry.npmjs.org")
        assert checks.foreign_registries({"package-lock.json": ok}) == []

    def test_a_removed_line_is_not_an_addition(self):
        """Only what the PR now depends on matters; a host it stopped using is
        not a finding."""
        removed = NPM.replace("+      \"resolved\": \"https://evil.example.com",
                              "-      \"resolved\": \"https://evil.example.com")
        assert checks.foreign_registries({"package-lock.json": removed}) == []


class TestASameVersionArtifactSwap:
    def test_an_integrity_change_with_no_version_change_is_reported(self):
        out = checks.integrity_without_version({"package-lock.json": REPUBLISH})
        assert len(out) == 1 and "1 artifact hash" in out[0]["title"]

    def test_a_real_version_bump_is_not(self):
        """The normal shape of dependency work, and the reason this check reads
        hunks rather than whole files: a bump moves both fields together."""
        assert checks.integrity_without_version({"package-lock.json": BUMP}) == []

    def test_two_packages_in_one_file_are_counted_once_each(self):
        out = checks.integrity_without_version(
            {"package-lock.json": REPUBLISH + REPUBLISH.replace("ok", "ok2")})
        assert "2 artifact hash(es)" in out[0]["title"]

    def test_hunks_do_not_bleed_into_each_other(self):
        """A version line in a DIFFERENT hunk must not excuse an artifact swap
        forty thousand lines away."""
        out = checks.integrity_without_version({"package-lock.json": BUMP + REPUBLISH})
        assert len(out) == 1 and "1 artifact hash" in out[0]["title"]


class TestNothingToSay:
    def test_no_lockfiles_is_silent(self):
        assert checks.lockfile_changes(None) == []
        assert checks.lockfile_changes({}) == []

    def test_an_ordinary_bump_is_silent(self):
        assert checks.lockfile_changes({"package-lock.json": BUMP}) == []


class TestItReachesTheReview:
    def test_pr_diff_keeps_the_lockfile_blob_it_skips(self, monkeypatch):
        """The blob is thrown away with the rest of the skipped text unless it
        is kept on purpose — and the model must still never see it."""
        raw = ("diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n"
               "@@\n+x\n"
               "diff --git a/package-lock.json b/package-lock.json\n"
               "--- a/package-lock.json\n+++ b/package-lock.json\n" + NPM)
        monkeypatch.setattr(pr, "gh", lambda *a, **k: raw)
        diff, _, skipped = pr.pr_diff("repo", 1)
        assert list(skipped) == ["package-lock.json"]
        assert "package-lock.json" in diff.lockfiles
        assert "evil.example.com" not in diff.full, "the model must not be shown it"

    def test_run_all_asks(self, tmp_path):
        out = checks.run_all(str(tmp_path), [], title="SCRUM-1 x",
                             lockfiles={"package-lock.json": NPM})
        assert any("not the usual registry" in f["title"] for f in out)

    def test_main_hands_the_lockfiles_to_the_checks(self, monkeypatch):
        """The seam, not the helper. `run_all` reading a lockfile is useless if
        `main` never passes one — and the last two tickets each spent a review
        round on exactly this kind of untested forwarding."""
        import json
        seen = {}
        d = pr._Diff("--- a/x\n+++ b/x\n@@\n+x\n")
        d.full = str(d)
        d.lockfiles = {"package-lock.json": NPM}
        monkeypatch.setattr(pr, "pr_diff", lambda *a: (d, [], pr._Skipped([])))
        monkeypatch.setattr(pr, "_already_reviewed", lambda *a, **k: "")
        monkeypatch.setattr(pr, "checkout", lambda *a: None)
        monkeypatch.setattr(pr, "build_context", lambda *a: "")
        monkeypatch.setattr(pr, "conversation", lambda *a: "")
        monkeypatch.setattr(pr, "changed_since_last_review", lambda *a, **k: "")
        monkeypatch.setattr(pr, "commit_messages", lambda *a: [])
        monkeypatch.setattr(pr.ctx, "expand_hunks", lambda d_, w, **k: d_)
        monkeypatch.setattr(pr, "review_findings", lambda *a, **k: [])
        monkeypatch.setattr(pr, "_revise", lambda f, w, r: (f, []))
        monkeypatch.setattr(pr.checks, "run_all",
                            lambda *a, **k: seen.setdefault(
                                "lockfiles", k.get("lockfiles")) and [])
        monkeypatch.setattr(pr, "post_review", lambda *a, **k: "COMMENT")
        monkeypatch.setattr(pr, "_pr_is_gone", lambda *a: None)
        monkeypatch.setattr(pr, "gh", lambda *a, **k: json.dumps(
            {"draft": False, "state": "open", "merged": False, "title": "SCRUM-1 x",
             "user": {"login": "someone"}, "head": {"sha": "a" * 40}}))
        monkeypatch.setattr(pr.sys, "argv", ["pr-review", "app", "7"])
        monkeypatch.delenv("DRY", raising=False)
        pr.main()
        assert seen["lockfiles"] == {"package-lock.json": NPM}


class TestEveryFormatIsBothSkippedAndChecked:
    """`LOCKFILE` listed seven formats and `SKIP` five, under a comment saying
    the two "cannot drift" — so `pnpm-lock.yaml`, `Cargo.lock` and
    `Gemfile.lock` were shown to the model (burning the budget SKIP exists to
    protect) AND never captured for the checks. Two hand-written lists are what
    drift; `SKIP` is built from the same names now."""

    NAMES = ("package-lock.json", "yarn.lock", "poetry.lock", "uv.lock",
             "bun.lockb", "pnpm-lock.yaml", "Cargo.lock", "Gemfile.lock")

    def test_a_checkable_lockfile_is_always_a_skipped_one(self):
        for name in self.NAMES:
            assert pr.SKIP.search(f"diff --git a/{name} b/{name}"), name

    def test_and_a_skipped_lockfile_is_always_checkable(self):
        for name in self.NAMES:
            assert pr.LOCKFILE.search(name), name

    def test_a_nested_path_counts_too(self):
        assert pr.LOCKFILE.search("services/api/Cargo.lock")
        assert not pr.LOCKFILE.search("Cargo.lock.bak")


class TestFormatsThatNameTheirRegistryDifferently:
    def test_cargos_index_is_not_a_foreign_host(self):
        """Cargo names crates.io as
        `registry+https://github.com/rust-lang/crates.io-index`, so a host check
        alone flags `github.com` on every Rust dependency — noise that would get
        this muted before it caught anything."""
        cargo = ('@@ -1,3 +1,4 @@\n'
                 '+source = "registry+https://github.com/rust-lang/crates.io-index"\n')
        assert checks.foreign_registries({"Cargo.lock": cargo}) == []

    def test_but_another_github_source_still_is(self):
        """The allowlist is the index URL, not the host: a git dependency on
        some other repository is exactly what this should surface."""
        cargo = ('@@ -1,3 +1,4 @@\n'
                 '+source = "git+https://github.com/someone/else"\n')
        assert len(checks.foreign_registries({"Cargo.lock": cargo})) == 1

    def test_gemfiles_remote_is_read(self):
        """Gemfile.lock names its registry with `remote:`, which is none of
        `resolved`/`url`/`source` — so a swapped host, the exact signal this
        check exists for, was invisible in that format."""
        gems = '@@ -1,3 +1,4 @@\n+  remote: https://gems.evil.example/\n'
        out = checks.foreign_registries({"Gemfile.lock": gems})
        assert len(out) == 1 and "gems.evil.example" in out[0]["detail"]

    def test_rubygems_itself_is_fine(self):
        gems = '@@ -1,3 +1,4 @@\n+  remote: https://rubygems.org/\n'
        assert checks.foreign_registries({"Gemfile.lock": gems}) == []
