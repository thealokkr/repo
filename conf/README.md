# Runtime configuration

`frozen_routing_policy.json` is the development-calibrated routing configuration used by the CLI.
Create it once with `cloudserve calibrate --development-input data/tickets/development_tickets.json`.
The `cloudserve run` command reads this same file by default, including for validation runs.
Keep the policy in version control after calibration; run results and logs belong under `outputs/`.
