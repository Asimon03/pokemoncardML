# Setup

Requires Python 3.10-3.12

From the project root, run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

To install requirements and enter the virtual env

# Usage

Test images are in `test_images/`

```bash
python src/test_embed.py
```

If you see that the first two embeddings are the same, and the last one is different, it is working.