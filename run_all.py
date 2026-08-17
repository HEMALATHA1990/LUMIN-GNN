import os

experiments = [
    "cora_gcn.py",
    "cora_gat.py",
    "cora_graphsage.py",
    "cora_h2gcn.py",
    "cora_mixhop.py",
    "cora_lumin_gnn.py",

    "citeseer_gcn.py",
    "citeseer_gat.py",
    "citeseer_graphsage.py",
    "citeseer_h2gcn.py",
    "citeseer_mixhop.py",
    "citeseer_lumin_gnn.py",

    "chameleon_gcn.py",
    "chameleon_gat.py",
    "chameleon_graphsage.py",
    "chameleon_h2gcn.py",
    "chameleon_mixhop.py",
    "chameleon_lumin_gnn.py",

    "arxiv_gcn.py",
    "arxiv_gat.py",
    "arxiv_graphsage.py",
    "arxiv_h2gcn.py",
    "arxiv_mixhop.py",
    "arxiv_lumin_gnn.py",
]

for exp in experiments:
    print("=" * 80)
    print("Running:", exp)
    print("=" * 80)
    os.system(f"python experiments/{exp}")