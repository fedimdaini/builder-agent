| Test case | Family | Retriever | Top 3 (id variant) | Same family in top 3 |
|---|---|---|---|---|
| gen-010 flask_2_0_3 | incompatible_pin | naive | gen-004 python_3_12; fault-001 mlflow_client_server_mismatch; gen-008 mlflow_client_unpinned | no |
| gen-010 flask_2_0_3 | incompatible_pin | advanced | gen-006 libffi8_removed; gen-004 python_3_12; gen-003 numpy_2_0_2 | yes (rank 3) |
| gen-001 mlflow_client_3_1_4 | version_mismatch | naive | gen-008 mlflow_client_unpinned; gen-007 mlflow_client_3_0_1; fault-001 mlflow_client_server_mismatch | yes (rank 1) |
| gen-001 mlflow_client_3_1_4 | version_mismatch | advanced | gen-007 mlflow_client_3_0_1; gen-008 mlflow_client_unpinned; fault-001 mlflow_client_server_mismatch | yes (rank 1) |
| gen-013 mlflow_uri_empty | env_var | naive | gen-008 mlflow_client_unpinned; gen-007 mlflow_client_3_0_1; gen-012 mlflow_uri_wrong_port | yes (rank 3) |
| gen-013 mlflow_uri_empty | env_var | advanced | gen-007 mlflow_client_3_0_1; gen-012 mlflow_uri_wrong_port; gen-008 mlflow_client_unpinned | yes (rank 2) |
| gen-011 python_3_13 | python_version | naive | gen-008 mlflow_client_unpinned; gen-009 xgboost_3_0_0; gen-007 mlflow_client_3_0_1 | no |
| gen-011 python_3_13 | python_version | advanced | gen-009 xgboost_3_0_0; gen-004 python_3_12; fault-001 mlflow_client_server_mismatch | yes (rank 2) |
naive: same family in top 3 for 2/4
advanced: same family in top 3 for 4/4
