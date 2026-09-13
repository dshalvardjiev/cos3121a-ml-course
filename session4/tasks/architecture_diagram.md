# The end-to-end ML application (annotate me)
```
 [S3: raw data] --> [Glue: prep jobs] --> [SageMaker training job] --> [Model registry]
                                                                            |
        [Application / CRM]  <--  [REST API (API GW + Lambda)]  <--  [SageMaker endpoint]
                 |                                                          ^
                 +---------------->  [Monitoring: latency, errors, drift] --+
```
For each box: which role owns it (DE / DS / MLE / MLOps)?
Pick ONE failure point: what breaks, what signal detects it, who gets paged?
