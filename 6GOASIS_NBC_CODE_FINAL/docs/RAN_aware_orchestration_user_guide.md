# Copyright 2026 Nearby Computing S.L.

**RAN-aware Edge Orchestration (NearbyBlock) — Functionality & User Guide**

Overview
- **What:** The NearbyOne orchestration NearbyBlock that implements RAN-aware placement and lifecycle management for the CTTC CARLA edge application.
- **Block location:** [block-charts/cttc-carla-block-chart-weightedQuery-placement](block-charts/cttc-carla-block-chart-weightedQuery-placement)

Key files and configuration
- Block Helm chart: `block-charts/cttc-carla-block-chart-weightedQuery-placement/Chart.yaml` and `values.yaml` — configure deployment variables, Prometheus queries and Kafka settings.
- Important `values.yaml` entries:
  - `placement.site.label`: site label for placement
  - `connections.prometheusQueries`: enable/disable Prometheus-based metrics and set `address` and `query.metric_name`
  - `koc.kafka.broker` and `koc.kafka.topic`: Kafka broker and topic used for migration events

How it works (short)
- The block collects RAN metrics (latency/throughput) via Prometheus queries, computes a weighted placement decision and emits migration events to the configured Kafka topic. It also contains configuration for the target chart (CTTC CARLA app) to be deployed on selected edge sites.

Deployment notes
- This block is intended to be consumed by the NearbyOne orchestration framework. To adapt or test locally:
  1. Ensure you have access to a NearbyOne instance and you have onboarded the block.
  2. Edit `values.yaml` to point `koc.kafka.broker` to your Kafka broker and set Prometheus `address`.
  3. Ensure the referenced application chart (`deployments.cttcCarla.configuration.chart`) points to a reachable registry or chart repository, or replace with a local chart path.

Verification
- Verify Prometheus connectivity by querying the configured `address` manually.
- Verify Kafka connectivity by publishing a test message to `koc.kafka.topic` and confirming the operator or consumers receive it.