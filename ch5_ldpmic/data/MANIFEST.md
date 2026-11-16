# Data manifest

Exact source URL and SHA256 of every downloaded file.  Filled in by hand from
the loader's first verified download (`LDPMIC_ALLOW_UNVERIFIED=1` prints the
digest); after that the loader refuses any file whose digest differs.
Nothing under `data/cache/` is committed.

Sizes after loading: adult 32561 train / 16281 test records, d = 105; bank 41188 records (32950 train / 8238 test), d = 65; mnist and fmnist 60000 train / 10000 test, d = 784 (torchvision also checks its own MD5 list).

| dataset | file | URL | SHA256 | downloaded (UTC) |
|---|---|---|---|---|
| adult | adult.data | https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data | `5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d` | 2026-10-26 21:46 |
| adult | adult.test | https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test | `a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05` | 2026-10-26 21:46 |
| adult | adult.names | https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names | `c248284c0b5de30c9e1958d6cdd168a34a654758b620e68f46aefa83fc0a576a` | 2026-10-26 21:46 |
| bank | bank+marketing.zip | https://archive.ics.uci.edu/static/public/222/bank+marketing.zip | `e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4` | 2026-10-26 21:46 |
| bank | bank-additional-full.csv | (inside the zip above, path bank-additional/bank-additional-full.csv) | `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8` | 2026-10-26 21:46 |
| mnist | train-images-idx3-ubyte.gz | https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz | `440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609` | 2026-10-27 04:14 |
| mnist | train-labels-idx1-ubyte.gz | https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz | `3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c` | 2026-10-27 04:14 |
| mnist | t10k-images-idx3-ubyte.gz | https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz | `8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6` | 2026-10-27 04:14 |
| mnist | t10k-labels-idx1-ubyte.gz | https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz | `f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6` | 2026-10-27 04:14 |
| fmnist | train-images-idx3-ubyte.gz | http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz | `3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84` | 2026-10-27 04:14 |
| fmnist | train-labels-idx1-ubyte.gz | http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz | `a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845` | 2026-10-27 04:14 |
| fmnist | t10k-images-idx3-ubyte.gz | http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz | `346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073` | 2026-10-27 04:14 |
| fmnist | t10k-labels-idx1-ubyte.gz | http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz | `67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5` | 2026-10-27 04:14 |

## kdd (UCI Census-Income (KDD))

| file | URL | SHA256 |
|---|---|---|
| census+income+kdd.zip | https://archive.ics.uci.edu/static/public/117/census+income+kdd.zip | 54fd206becffcfaf099544c3938c681d64a65709800c94c00a4fba0a00df10c9 |
| census.tar.gz (inside the zip) | — | dd99f3edfcb6a29278e1d5a9dc18750772f3572f9ffe82844892b9f8b4902a4d |
| census-income.data (inside the tar) | — | 3676a81db7d3528f3f8b9f3c699d0f0aa28db45e6e994fa0b8ed38327539ee86 |
| census-income.test (inside the tar) | — | 98402b1ab879573d0a7f38a699a40258080e25e33d3401e7bf9c96d3fa0fab8c |
| census-income.names (inside the tar) | — | d6fea52977f68504ab3149351980458e09bfc3116808da8986646710d254ef98 |

## cdc (UCI CDC Diabetes Health Indicators, BRFSS 2015; DECISIONS.md D58)

| file | URL | SHA256 |
|---|---|---|
| data.csv | https://archive.ics.uci.edu/static/public/891/data.csv | 9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964 |
