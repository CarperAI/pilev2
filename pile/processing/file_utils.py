from lm_dataformat import Reader, Archive
from datasets import load_dataset, load_from_disk
from pathlib import Path
import multiprocessing as mp
import tqdm
class DataConverter:
    def __init__(self, data_dir, output_dir, input_format="parquet"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.format = input_format

    def convert_to_lm_dataformat(self):
        data_dir = Path(self.data_dir)
        text_column = "text" if "the_stack" not in self.data_dir else "content"

        if self.format == "parquet":
            parquets = [ str(f) for f in list(data_dir.glob("*.parquet"))]
            
            ds = load_dataset("parquet", data_files = {"train": parquets})['train']
            self._convert(ds, text_column)
        elif self.format == "jsonl":
            jsonls = list(data_dir.glob("*.jsonl"))
            ds = load_dataset("json", data_files = {"train": jsonls})['train']
            self._convert(ds, text_column)
        elif self.format == "arrow":

            ds = load_from_disk(data_dir)
            self._convert(ds, text_column)

    """
    Converts a huggingface dataset to lm_dataformat
    """
    def _convert(self, hf_ds, text_column="text"):
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ar = Archive(str(self.output_dir))

        for i, row in tqdm.tqdm(enumerate(hf_ds), total=len(hf_ds), desc=f"Converting {self.data_dir} to lm_dataformat"):
            ar.add_data (row [text_column], meta=row ['meta'])
        
        ar.commit()
               
                
    