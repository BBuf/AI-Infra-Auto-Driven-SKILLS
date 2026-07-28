# Clean-Room Publication Policy

## Contents

- [Publication Boundary](#publication-boundary)
- [Allowed Evidence](#allowed-evidence)
- [Forbidden Data](#forbidden-data)
- [Internal-to-Public Mapping](#internal-to-public-mapping)
- [Performance Claims](#performance-claims)
- [Open, Closed, Reverted, and Experimental Work](#open-closed-reverted-and-experimental-work)
- [Final Scan](#final-scan)

## Publication Boundary

Start the public bundle from a new evidence ledger containing public upstream
URLs and public source paths. Treat private development history as read-only
human context outside the bundle.

The boundary covers:

- skill references and examples;
- generated Day-0 bundle files;
- source comments, tests, docs, and cookbook cells;
- commit messages and PR title/body;
- pasted logs, commands, screenshots, and benchmark tables.

## Allowed Evidence

- public `sgl-project/sglang` PRs, issues, source, tests, docs, releases, and CI;
- public model configuration, weights, processor, and technical report;
- public dependency documentation and releases;
- synthetic examples that contain no unreleased model or environment data;
- generic engineering guardrails stated without a private provenance claim.

## Forbidden Data

Do not publish:

- private repository, PR, issue, branch, tag, or commit identifiers;
- private authorship, review assignments, messages, or organizational metadata;
- machine names, usernames, IPs, ports, jump hosts, or filesystem paths;
- private registries, image tags/digests, object-store URIs, model revisions, or
  artifact hashes;
- internal benchmark round IDs, trace names, run directories, or dashboards;
- secrets, credentials, tokens, signed URLs, or environment dumps;
- performance data lacking a public source.

Do not transform a forbidden identifier into a recognizable abbreviation.

## Internal-to-Public Mapping

Maintain any mapping only in memory or an uncommitted local file.

For each private lesson:

1. identify the mechanism without copying identifiers;
2. find public corroboration in an upstream PR, code path, test, or document;
3. cite the public evidence and its exact state;
4. omit private chronology and attribution;
5. if public corroboration is absent, express only a generic guardrail or omit
   the lesson entirely.

Never commit the mapping, its counts, or the denylist.

## Performance Claims

Publish a number only when its public source identifies:

- source/model revision;
- hardware and node/GPU count;
- precision and backend;
- parallel topology;
- input/output or request distribution;
- concurrency or batch shape;
- metric definition.

Keep campaign-level attribution, microbenchmark speedup, and end-to-end
throughput separate. Do not generalize one shape or GPU to all deployments.

## Open, Closed, Reverted, and Experimental Work

- Mark open PRs with capture date and immutable head.
- Treat closed-unmerged work as a rejected or superseded approach.
- When merged work is later reverted, present the final reachable state and
  retain the earlier work only as a negative lesson.
- Treat default-disabled, unreachable, or shape-uncovered fast paths as
  conditional, not shipped universal behavior.

## Final Scan

Before every public commit and PR update:

1. validate all eight bundle files;
2. scan with an uncommitted denylist containing exact forbidden literals;
3. scan added lines for absolute work paths, IPs, SSH remotes, secret-like
   prefixes, and non-public GitHub repositories;
4. inspect links, commands, code comments, fixture strings, and commit messages;
5. recheck live state for every open PR;
6. review the staged diff manually.

Fail closed. Remove or replace a questionable detail instead of guessing that
it is safe.
