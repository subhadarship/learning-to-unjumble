# Notes

The files in `data dir from DeepLearningExamples @b88db70`
were taken from [here](https://github.com/NVIDIA/DeepLearningExamples/tree/b88db70dc14952bd23a6f3467cae490809d74467/PyTorch/LanguageModeling/BERT/data)

Files edited are:
- `bertPrep.py`
- `WikiDownloader.py`

`create_datasets_from_wikidump.sh`: run this to download and extract wikidump data:

```text
./create_datasets_from_wikidump.sh | tee wikidump.log
```

`wikiextractor` is added a submodule separately
