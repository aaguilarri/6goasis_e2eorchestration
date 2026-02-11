# Copyright 2026 Nearby Computing S.L.

**Kafka Notify Operator — Functionality & User Guide**

Overview
- **What:** Kubernetes operator (Go) that emits/handles Kafka notifications used by the NearbyOne orchestration.
- **Code location:** [kafkanotify-operator](kafkanotify-operator)
- **Helm chart:** [helm-charts/kafka-notify-operator-helm-chart](helm-charts/kafka-notify-operator-helm-chart)

Build & deploy (quick)
1. Build and publish container image (adjust `IMG`):

```bash
make docker-build docker-push IMG=<registry>/kafkanotify-operator:tag
```

2. Install CRDs (if needed):

```bash
make install
```

3. Deploy controller using the image you published:

```bash
make deploy IMG=<registry>/kafkanotify-operator:tag
```

4. Apply sample Custom Resources:

```bash
kubectl apply -k config/samples/
```

Helm deployment option
- Use the Helm chart under `helm-charts/kafka-notify-operator-helm-chart` to install with Helm:

```bash
helm install kafkanotify ./helm-charts/kafka-notify-operator-helm-chart --set controllerManager.container.image.repository=<registry>/kafkanotify-operator --set controllerManager.container.image.tag=tag
```

Configuration notes
- The chart `values.yaml` contains the default image repository, RBAC, and CRD handling (`crd.enable`/`crd.keep`). Update `controllerManager.container.image.repository` and `tag` for your release.

Verification
- After deployment, confirm the controller pod is Running and check logs for connection to Kafka and any events processed.
