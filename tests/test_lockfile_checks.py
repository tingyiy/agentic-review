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

    def _drive(self, monkeypatch, diff_text, lockfiles, excluded=()):
        """`main` with the model stubbed out, returning what got POSTED."""
        import json
        seen = {}
        d = pr._Diff(diff_text)
        d.full = diff_text
        d.lockfiles = lockfiles
        monkeypatch.setattr(pr, "pr_diff",
                            lambda *a: (d, list(excluded), pr._Skipped(
                                list(lockfiles))))
        monkeypatch.setattr(pr, "_already_reviewed", lambda *a, **k: "")
        monkeypatch.setattr(pr, "checkout", lambda *a: None)
        monkeypatch.setattr(pr, "build_context", lambda *a: "")
        monkeypatch.setattr(pr, "conversation", lambda *a: "")
        monkeypatch.setattr(pr, "changed_since_last_review", lambda *a, **k: "")
        monkeypatch.setattr(pr, "commit_messages", lambda *a: [])
        monkeypatch.setattr(pr.ctx, "expand_hunks", lambda d_, w, **k: d_)
        monkeypatch.setattr(pr.ctx, "skeletons", lambda *a: "")
        monkeypatch.setattr(pr, "review_findings", lambda *a, **k: [])
        monkeypatch.setattr(pr, "_revise", lambda f, w, r: (f, []))
        monkeypatch.setattr(pr, "post_review",
                            lambda repo, n, ev, body, **k: (
                                seen.update(event=ev, body=body), ev)[1])
        monkeypatch.setattr(pr.status, "done", lambda *a: None)
        monkeypatch.setattr(pr.status, "nothing_to_review",
                            lambda repo, sha, why: seen.setdefault("quiet", why))
        monkeypatch.setattr(pr, "_pr_is_gone", lambda *a: None)
        monkeypatch.setattr(pr, "gh", lambda *a, **k: json.dumps(
            {"draft": False, "state": "open", "merged": False, "title": "SCRUM-1 x",
             "user": {"login": "someone"}, "head": {"sha": "a" * 40}}))
        monkeypatch.setattr(pr.sys, "argv", ["pr-review", "app", "7"])
        monkeypatch.delenv("DRY", raising=False)
        pr.main()
        return seen

    def test_a_lockfile_only_pr_is_still_reviewed(self, monkeypatch):
        """THE MOTIVATING CASE. caeli-marketing#243 changed an image and a
        `package-lock.json`: every file skipped, so `diff` is empty and the
        early return fires — before the checks. The feature would never have
        run on the pull request it was written for."""
        seen = self._drive(monkeypatch, "", {"package-lock.json": NPM})
        assert "quiet" not in seen, "took the nothing-to-review exit"
        assert "not the usual registry" in seen["body"]

    def test_a_clean_lockfile_only_pr_is_still_quiet(self, monkeypatch):
        """The other half: an ordinary dependency bump says nothing, and a
        comment on every dependabot PR is what gets a check muted."""
        seen = self._drive(monkeypatch, "", {"package-lock.json": BUMP})
        assert "quiet" in seen and "body" not in seen

    def test_an_oversized_file_does_not_swallow_the_lockfile_finding(
            self, monkeypatch):
        """`if excluded:` returned before `if lock_findings:`, so a PR with an
        over-ceiling file AND a dependency warning posted only the "too large"
        note. One body carries both now."""
        seen = self._drive(monkeypatch, "", {"package-lock.json": NPM},
                           excluded=["data/huge.jsonl"])
        assert "not the usual registry" in seen["body"], "the warning was dropped"
        assert "NOT opened" in seen["body"], "and the caveat must survive too"

    def test_a_normal_pr_reports_them_once(self, monkeypatch):
        """Computed before the early return AND passed to `run_all` would
        double every lockfile finding on a PR that has other changes."""
        seen = self._drive(monkeypatch, "--- a/x\n+++ b/x\n@@\n+x\n",
                           {"package-lock.json": NPM})
        assert seen["body"].count("not the usual registry") == 1


class TestUrlIsNotAlwaysAnArtifact:
    """npm's lockfile carries `"funding": {"url": "https://github.com/sponsors/…"}`
    on ordinary packages, and `url` is one of the fields this check reads — so
    every funded dependency flagged `github.com`. That false positive is what
    gets a check muted before it catches anything."""

    def test_a_funding_link_is_not_a_package_source(self):
        funding = '@@ -1,3 +1,4 @@\n+      "url": "https://github.com/sponsors/foo"\n'
        assert checks.foreign_registries({"package-lock.json": funding}) == []

    def test_a_wheel_url_still_is(self):
        """uv and poetry name the artifact with a bare `url`, and it IS a
        package location — the value says which."""
        wheel = ('@@ -1,3 +1,4 @@\n'
                 '+url = "https://evil.example/foo-1.0-py3-none-any.whl"\n')
        assert len(checks.foreign_registries({"uv.lock": wheel})) == 1

    def test_resolved_and_remote_need_no_suffix(self):
        """`resolved`, `source` and `remote` never name anything but a package
        location, whatever the value looks like."""
        gems = '@@ -1,3 +1,4 @@\n+  remote: https://gems.evil.example/\n'
        assert len(checks.foreign_registries({"Gemfile.lock": gems})) == 1


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
        cargo = '@@ -1,3 +1,4 @@\n+source = "git+https://github.com/someone/else"\n'
        assert len(checks.foreign_registries({"Cargo.lock": cargo})) == 1

    def test_rubygems_itself_is_fine(self):
        gems = '@@ -1,3 +1,4 @@\n+  remote: https://rubygems.org/\n'
        assert checks.foreign_registries({"Gemfile.lock": gems}) == []


class TestTheOtherTwoExits:
    """Both found by the reviewer on this PR: a second posting path that never
    passed the DRY guard, and a fingerprint that could not see the file the
    whole feature is about."""

    def test_dry_prints_the_lockfile_review_instead_of_posting(self, monkeypatch):
        """DRY's contract is that it prints what it WOULD do. The lockfile-only
        path posted anyway — a second exit reaching GitHub without passing the
        one guard."""
        import json
        seen = {}
        d = pr._Diff("")
        d.full = ""
        d.lockfiles = {"package-lock.json": NPM}
        monkeypatch.setattr(pr, "pr_diff",
                            lambda *a: (d, [], pr._Skipped(["package-lock.json"])))
        monkeypatch.setattr(pr, "_already_reviewed", lambda *a, **k: "")
        monkeypatch.setattr(pr, "_pr_is_gone", lambda *a: None)
        monkeypatch.setattr(pr, "post_review",
                            lambda *a, **k: seen.setdefault("posted", True) or "COMMENT")
        monkeypatch.setattr(pr.status, "done", lambda *a: None)
        monkeypatch.setattr(pr, "gh", lambda *a, **k: json.dumps(
            {"draft": False, "state": "open", "merged": False, "title": "SCRUM-1 x",
             "user": {"login": "someone"}, "head": {"sha": "a" * 40}}))
        monkeypatch.setattr(pr.sys, "argv", ["pr-review", "app", "7"])
        monkeypatch.setenv("DRY", "1")
        pr.main()
        assert "posted" not in seen, "DRY reached GitHub"

    def test_a_lockfile_change_moves_the_fingerprint(self):
        """`full` deliberately excludes the lockfile — 40,000 lines of JSON in
        the cross-reference names would drown them — but the mark in the body
        decides "has anything changed since the last review", and a
        lockfile-only push on top of an already-reviewed change IS a change."""
        source = "--- a/x\n+++ b/x\n@@\n+x\n"
        assert pr._diff_fp(source + NPM) != pr._diff_fp(source)

    def test_main_fingerprints_the_lockfile_too(self, monkeypatch):
        """The wiring, not the helper: `_already_reviewed` has to RECEIVE a
        diff that carries the lockfile, or a lockfile-only push on top of an
        already-reviewed change is skipped and its findings thrown away."""
        import json
        seen = {}
        d = pr._Diff("--- a/x\n+++ b/x\n@@\n+x\n")
        d.full = "--- a/x\n+++ b/x\n@@\n+x\n"
        d.lockfiles = {"package-lock.json": NPM}
        monkeypatch.setattr(pr, "pr_diff",
                            lambda *a: (d, [], pr._Skipped(["package-lock.json"])))
        monkeypatch.setattr(pr, "_already_reviewed",
                            lambda *a, **k: seen.setdefault("diff", a[3]) and "stop")
        monkeypatch.setattr(pr, "_pr_is_gone", lambda *a: None)
        monkeypatch.setattr(pr, "gh", lambda *a, **k: json.dumps(
            {"draft": False, "state": "open", "merged": False, "title": "SCRUM-1 x",
             "user": {"login": "someone"}, "head": {"sha": "a" * 40}}))
        monkeypatch.setattr(pr.sys, "argv", ["pr-review", "app", "7"])
        monkeypatch.delenv("DRY", raising=False)
        pr.main()
        assert "evil.example.com" in seen["diff"], "the lockfile is not fingerprinted"


    def test_a_lockfile_only_pr_consults_the_nothing_new_guard(self, monkeypatch):
        """It used to return before the guard, so the same finding was posted
        again on every re-request and every comment — the noise the guard
        exists to prevent, on the very PR shape this feature targets."""
        import json
        seen = {}
        d = pr._Diff("")
        d.full = ""
        d.lockfiles = {"package-lock.json": NPM}
        monkeypatch.setattr(pr, "pr_diff",
                            lambda *a: (d, [], pr._Skipped(["package-lock.json"])))
        monkeypatch.setattr(pr, "_already_reviewed",
                            lambda *a, **k: "this exact commit already has a review")
        monkeypatch.setattr(pr, "_pr_is_gone", lambda *a: None)
        monkeypatch.setattr(pr, "post_review",
                            lambda *a, **k: seen.setdefault("posted", True) or "COMMENT")
        monkeypatch.setattr(pr.status, "done", lambda *a: None)
        monkeypatch.setattr(pr.status, "nothing_to_review", lambda *a: None)
        monkeypatch.setattr(pr, "gh", lambda *a, **k: json.dumps(
            {"draft": False, "state": "open", "merged": False, "title": "SCRUM-1 x",
             "user": {"login": "someone"}, "head": {"sha": "a" * 40}}))
        monkeypatch.setattr(pr.sys, "argv", ["pr-review", "app", "7"])
        monkeypatch.delenv("DRY", raising=False)
        pr.main()
        assert "posted" not in seen, "re-posted a finding nothing had changed about"
