# Tests

This directory is reserved for lightweight tests that can run without long GPU
jobs. The current executable smoke suite is:

```bash
bash scripts/run_all_smoke_tests.sh
```

As reusable code is promoted into `decodeshare/`, add unit tests here for
artifact readers, config parsing, subspace utilities, and table formatting.
