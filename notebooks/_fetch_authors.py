"""Download and inspect the authors parquet."""
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

path = hf_hub_download(
    repo_id="storytracer/openlibrary_dump_2024-04-30",
    filename="data/parquet/ol_dump_authors_2024-04-30.parquet",
    repo_type="dataset",
)
print(f"Authors downloaded to: {path}")

pf = pq.ParquetFile(path)
print(f"Authors: {pf.metadata.num_rows:,} rows")
print(f"Schema:\n{pf.schema_arrow}")

t = pf.read_row_groups([0])
df = t.to_pandas().head(5)
for i in range(min(5, len(df))):
    r = df.iloc[i]
    print(f"  key={r['key']}, name={r['name']}")
