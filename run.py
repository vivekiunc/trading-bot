import sys, os
sys.path.append(os.path.abspath("src"))

import universe as u
import features as ft

# stage 1 — run once, comment out after bars.parquet exists
assets = u.get_asset_universe()
bars = u.download_bars(assets.symbol.tolist())
univ = u.build_asof_universe(bars, assets)

# stage 2
cols = ["mom_12_1", "reversal_1m", "vol_3m", "beta", "illiq"]
feats = ft.build_features(bars, univ)
feats = ft.cross_sectional_rank(feats, cols)
feats.to_parquet("data/features.parquet")
print(feats.groupby("date").size().describe())