# Fraud Detection

![Version: 0.2.1](https://img.shields.io/badge/Version-0.2.1-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: 1.16.0](https://img.shields.io/badge/AppVersion-1.16.0-informational?style=flat-square)

A Helm chart for Kubernetes

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| dataset.local.dataset_path | string | `"nil"` | Host Path to Dataset Directory |
| dataset.local.gnn_path | string | `"nil"` | Host Path to GNN Output |
| dataset.local.preprocess_path | string | `"nil"` | Host Path to Preprocessing Output |
| dataset.nfs.path | string | `"nil"` | Path to Local NFS Share in Cluster Host |
| dataset.nfs.server | string | `"nil"` | Hostname of NFS Server |
| dataset.nfs.subPath | string | `"nil"` | Path to dataset in Local NFS |
| dataset.s3.key | string | `"nil"` | Path to Dataset in S3 Bucket |
| dataset.type | string | `"<nfs/s3/local>"` | `nfs`, `s3`, or `local` dataset input enabler |
| image.name | string | `"intel/ai-workflows"` |  |
| metadata.name | string | `"fraud-detection"` |  |
| proxy | string | `"nil"` |  |
