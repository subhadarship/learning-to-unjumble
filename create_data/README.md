# Notes

The files in `data dir from DeepLearningExamples @b88db70`
were taken from [here](https://github.com/NVIDIA/DeepLearningExamples/tree/b88db70dc14952bd23a6f3467cae490809d74467/PyTorch/LanguageModeling/BERT/data)

Files edited are:
- `bertPrep.py`
- `WikiDownloader.py`

`create_datasets_from_wikidump.sh`: run this to download and extract wikidump data:

```shell script
./create_datasets_from_wikidump.sh | tee log_wikidump.txt
```

During the above run, following INFO was printed which may be useful
```shell script
INFO: Finished 4-process extraction of 15016 articles in 458.9s (32.7 art/s)
INFO: total of page: 15111, total of articl page: 15016; total of used articl page: 15016
```

`wikiextractor` is added as a submodule separately
