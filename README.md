# PythonIndexer
## Problem
Research labs often have Python scripts scattered all over different lab data and experiment files. When someone wants to find an old file, they might remember a certain function, comment, or variable name but not the exact filename. In large filesystems, a more sophisticated Python file indexer is needed.

## Description
- Robust searching/indexing of Python files in large file systems
- Save indices for use later or for sharing
- Load large indices in seconds
- Robust index search across code, comments, docstrings, variable names
- Further search customization with normalized weighting by file size and natural language matching

## Real-Life Usage
PythonIndexer is used in the [Marto Lab](https://martolab.dana-farber.org/) to index 200+ Python scripts. It indexes 150 TB of data in thirty minutes. Whenever someone needs to search for a script, they can search the load the index and perform a full natural language search in seconds.
