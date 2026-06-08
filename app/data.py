"""
Här sparar jag datasetet efter att användaren laddat upp en csv fil,
det ligger kvar i minnet så det kan användas av /data/stats och /ai/ask.
För kk2 räcker det att det sparas i minnet.
"""

import pandas as pd

dataset: pd.DataFrame | None = None