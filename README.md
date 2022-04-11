### Setup
Python 3.8
```
conda create graph-factuality -n python=3.8
pip install -r requirements.txt
pip install -r requirements.dev.txt
```

### Lint & formatting
```
black models/
flake8 models/
```

#### Generate baseline summaries
```
python scripts/baseline_xsum_summaries.py
```

Summaries are stored in [data/xsum/](`data/xsum/`)