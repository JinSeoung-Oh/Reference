## From https://medium.com/@pymupdf/extracting-text-from-multi-column-pages-a-practical-pymupdf-guide-a5848e5899fe

import pymupdf4llm
import pathlib
import sys
  
  
filename = sys.argv[1]  # read filename from command line
outname = filename.replace(".pdf", ".md")
md_text = pymupdf4llm.to_markdown(filename)
  
# output document markdown text as one string
pathlib.Path(outname).write_bytes(md_text.encode())
