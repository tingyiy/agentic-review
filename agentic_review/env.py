"""Credentials, from the environment or from a file on the box.

A GitHub Actions step inherits almost nothing from the machine it runs on — on a
self-hosted runner it gets `LANG` and little else. So a job driven from OUTSIDE
a shell that sourced the credentials has to be told where they live, and the
difference used to be discovered one variable at a time, at the moment each
variable was needed.

`REVIEW_ENV_FILES` is a colon-separated list of `KEY=value` files, tried in
order. Empty by default: an adopter passes real environment variables and never
touches this.
"""
import os

FILES = [os.path.expanduser(p)
         for p in os.environ.get("REVIEW_ENV_FILES", "").split(":") if p]


def get(key, *extra_files):
    """`os.environ[key]`, else the first file that defines it, else None.

    A found value is EXPORTED, so a subprocess (git, an alert command) sees it
    too without every caller remembering to pass it down.
    """
    value = os.environ.get(key)
    if value:
        return value
    for path in list(extra_files) + FILES:
        try:
            with open(os.path.expanduser(path)) as fh:
                for line in fh:
                    if not line.startswith(f"{key}="):
                        continue
                    # Strip an inline comment and surrounding quotes. No secret
                    # read here can contain `#` — API tokens are restricted
                    # alphabets — and a `.env` written by hand is far more
                    # likely to carry a trailing note.
                    value = line.split("=", 1)[1].split("#")[0].strip()
                    value = value.strip('"').strip("'")
                    if value:
                        os.environ[key] = value
                        return value
        except OSError:
            continue
    return None
