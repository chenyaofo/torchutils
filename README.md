# PyTorch Utilities for Private Usage

author: chenyaofo@gmail.com

The library is designed for integration in the main project.

``` bash
git clone --depth 1 https://github.com/chenyaofo/torchutils.git torchutils_repo && mv torchutils_repo/torchutils . && rm -rf torchutils_repo
```


## Changelogs

### v0.2.0

 - Add `torchutils.snmp_launch` to quickly start Single-Node multi-process distributed training. 

### v0.1.3

 - Add diagnostic infos to output directory automatically

### v0.1.2

 - Add support to run in OpenPAI: automatically detect whether runing in OpenPAI, copy env variables from OpenPAI to init distributed mode

### v0.1.1

 - Add three utility function: get_branch_name, get_gpus_memory_info, get_free_port

### v0.1.0

 - Automatically initialize distributed mode according to env variables
 - Out-of-the-box logger and tensorboard summary writer
 - Metrics supported distributed mode, including Accuracy, ConfusionMatrix, AverageMetric
 - Common utilities, such as compute_flops, compute_nparams ...
 - Integrate typed_args library from https://github.com/SunDoge/typed-args/tree/v0.4