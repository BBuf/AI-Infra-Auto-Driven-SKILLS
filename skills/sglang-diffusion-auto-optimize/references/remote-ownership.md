# Remote Campaign Ownership

## Responsibility Boundary

The host skill owns:

- SSH/jumpbox access;
- container entry;
- repository and persistent-storage paths;
- GPU allocation and process-ownership rules; and
- host-specific cleanup.

The Python controller owns:

- immutable source and workload locks;
- baseline, profiling, search, integration, and correctness state;
- executor process receipts and leases;
- watchdog restart behavior;
- token and progress ledgers; and
- terminal patch/certificate packaging.

The conversational skill coordinates the two. Do not encode private hostnames,
addresses, credentials, or container IDs into the public controller.

## Launch Safety

Before launch:

1. inspect GPU processes and available memory;
2. verify the SGLang origin and fetched main commit;
3. preserve dirty user work;
4. validate the benchmark entrypoint and five-prompt file;
5. make the campaign root persistent; and
6. inherit credentials through the environment.

After launch:

1. record campaign ID/path and watchdog PID;
2. confirm the watchdog receipt and first heartbeat;
3. use only campaign-recorded process groups for restart/cleanup; and
4. leave evidence available after a terminal result.

Do not use broad `pkill`, recursive deletion, `git reset --hard`, or worktree
cleanup outside the campaign-owned paths.
