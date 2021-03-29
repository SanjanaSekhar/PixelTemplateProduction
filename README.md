# Pixel Template Production
Code for production of pixel templates for CMS

## Compiling

Everything is run standalone from the cmssw environment but makes use of some of the code in
it. 


Run the script **fetch\_cmssw\_code.sh** grabs the latest version of all the needed pixel code from the [cmssw github](https://github.com/cms-sw/cmssw).
If you want to grab from a branch other than the cmssw master (eg to test some
changes), you can change the `branch` variable in the script to point to a different branch.

The code shared with CMSSW uses the [vdt](https://github.com/dpiparo/vdt) library so you must have it installed. 
For instructions to install vdt see [their github](https://github.com/dpiparo/vdt). (If the CMake version is old, hacky fix: change it in CMakeLists.txt.)
The Makefile assumes it installed to /usr/local/, if it is installed to some
other location, change the `vdt_dir` variable in the Makefile to point to the correct location. (Better not to use `/usr/local` if working on LPC because one does not have root access.)

You may also need to install BOOST if working locally: `sudo apt-get install libboost-all-dev`

All of the source code is in the src/ directory which contains a Makefile. So you should be able to compile by simply changing to the src/ directory and running `make`. All the compiled executables are put in the bin/ directory. (To obtain standalone ROOT C++ on LPC, check [this]( https://uscms.org/uscms_at_work/computing/setup/setup_software.shtml#lcgsoft))


Note that this can be compiled and run from within a CMSSW release by linking to
the compiled vdt and BOOST used in CMSSW. Eg on lxplus:

> vdt_dir=/cvmfs/cms.cern.ch/slc7_amd64_gcc820/cms/vdt/0.4.0-nmpfii/include
> includes= -I. -I../cmssw_code/ -I$(vdt_dir) -I/cvmfs/cms.cern.ch/slc7_amd64_gcc820/external/boost/1.67.0-pafccj/include





## Making Templates

There are two simple bash scripts that run the necessary executables to make templates: **make\_1d\_templates.sh** and **make\_2d\_templates.sh**. 

They should be run inside a directory containing pixelav events and a config file named `pix_2t.proc`. They take 1 argument, which is the location of the bin/ directory. 

Many files are produced when making templates but for the 1d production the real output is one template file named `template_summary_zpXXXX.out`and one gen errors file named `generror_summary_zpXXXX.out`. For 2d production it is one 2d template file named `template_summary2D_zpXXXX.out`. (The XXXX will be the starting file number in your config). 

The format `pix_2t.proc` is as follows:

> start\_file nfiles noise thresh1 thresh2 thresh1\_noise_frac common\_noise\_frac gain\_noise\_frac readout\_noise frontend\_type

> use\_l1\_offset write\_header xtalk\_frac xtalk\_spread


> id NTy NTyx NTxx DType Bfield VBias temp fluenc qscale 

Note that NTy is not used by the 2D templates so its value doesn't matter, but to keep the format consistent something must be there. 
Using 0 for xtalk\_frac will turn off cross talk. 
Extra parameters on any of the lines will be ignored. 

An example config for 1D barrel templates is: 

> 58401 205 250. 1600. 1600. 0.073 0.080 0.080 350. 0

> 0 1 0.0 0.0

> 900 60 5 29 0 3.8 125. 263. 0. 1.




## Description of Executables
**gen\_xy\_template** : Makes 1D projections of average charge distributions from pixelav events. Also does charge variance fits. Makes files like `ptemp_XXXX.txt` and `ztemp_XXXX.txt`.

**gen\_xy\_template2d** : Makes 2d projections of average charge distributions from pixelav events.  Makes files like `zptemp_XXXX.txt`.

**gen\_zp\_template**: Uses the 1D projections and pixelav events. Runs the generic and template reco (using CMSSW code) on pixelav events to get resolution different algorithms and compute corrections to be saved in templates. Outputs one template file named `template_summary_zpXXXX.out`and one gen errors file named `generror_summary_zpXXXX.out`.

**gen\_zp\_template2d**: Uses the 1D and 2D projections and pixelav events. Runs the 2D template reco (using CMSSW code) on pixelav events to get resolution. Outputs one 2d template file named `template_summary2D_zpXXXX.out`.

**compare\_templates**: Takes in the file names of two templates and checks that all numerical values are the same within some threshold (default is 10^-5). It lists any discrepancies with the line number for investigation. Useful for testing changes. 

**test_template**: Uses pre-made 1D templates to run local version of CMSSW 1D template reco and makes various plots. Useful for testing a new set of 1D templates. 
Should be run a directory with template\_events, generror and template\_summary files. Also takes a config called `test_params.txt`.
The first line of the config is the same as the `pix_2t.proc` but without the
nfiles parameter (because it will only use one file). The second line has three
parameters, the first is the file number of the template (the XXXXX) and the
second is the `use_l1_offset` parameter, the third is the cross talk fraction. 

An example `test_params.txt` config is:
> 58606 150. 1600. 1600. 0.073 0.080 0.08 350. 0

> 58401 0 0.0 0.0

## Installing the Tensorflow C++ API

[This link](https://gist.github.com/kmhofmann/e368a2ebba05f807fa1a90b3bf9a1e03) was complete, but is specific to the distributions mentioned. Please check with your distributions to see if it is compatible.

There are a few modifications to the steps in the link, they are detailed here:

- After compiling Bazel, remember to add `bazel-3.1.0/output/` to your PATH
- Before your run `./configure`, you should export a few flags to tell TensorFlow whether it should build certain features. In particular, `export TF_NEED_CUDA=0` disables GPU support. If you need GPU support, you should do the steps in the guide related to CUDA. You can run `./configure` once you defined these flags. This is the list of flags used in the CMSSW TF setup:

```
export TF_NEED_JEMALLOC=0
export TF_NEED_HDFS=0
export TF_NEED_GCP=0
export TF_ENABLE_XLA=0
export TF_NEED_OPENCL=0
export TF_NEED_CUDA=0
export TF_NEED_VERBS=0
export TF_NEED_MKL=0
export TF_NEED_MPI=0
export TF_NEED_S3=0
export TF_NEED_GDR=0
export TF_NEED_OPENCL_SYCL=0
export TF_SET_ANDROID_WORKSPACE=false
export TF_NEED_KAFKA=false
export TF_NEED_AWS=0
export TF_DOWNLOAD_CLANG=0
export TF_NEED_IGNITE=0
export TF_NEED_ROCM=0
export TF_NEED_TENSORRT=0
```

- When you run `./configure` read the options presented carefully. Do not change the bazel configurations options. 
- Replace `bazel build --config=opt -c opt //tensorflow/tools/pip_package:build_pip_package` with the following:
```
export BAZEL_OPTS="--config=opt -c opt -s --verbose_failures -j 8 --config=noaws --config=nogcp --config=nohdfs --config=nonccl"
bazel build $BAZEL_OPTS //tensorflow/tools/pip_package:build_pip_package
```
- You may get an error saying that `/tmp/tensorflow_pkg/tensorflow-2.3.0-cp38-cp38-linux_x86_64.whl`. In such a case, recreate the wheel file in a different directory (within `tensorflow/` for example). If you obtain warnings saying certain header files are not found, you can ignore them.
- After testing the Python installation we can build targets for the C++ implementation.

```
bazel build $BAZEL_OPTS //tensorflow:tensorflow
bazel build $BAZEL_OPTS //tensorflow:tensorflow_cc
bazel build $BAZEL_OPTS //tensorflow/tools/graph_transforms:transform_graph
bazel build $BAZEL_OPTS //tensorflow/compiler/tf2xla:tf2xla
bazel build $BAZEL_OPTS //tensorflow/compiler/xla:cpu_function_runtime
bazel build $BAZEL_OPTS //tensorflow/compiler/xla:executable_run_options
bazel build $BAZEL_OPTS //tensorflow/compiler/tf2xla:xla_compiled_cpu_function
bazel build $BAZEL_OPTS //tensorflow/core/profiler
bazel build $BAZEL_OPTS //tensorflow:install_headers
```

## Compiling and running 

Activate the virtual environment that contains tensorflow:
```
source ~/.virtualenvs/tf_dev/bin/activate
```
This can later be deactivated with `deactivate`.

Obtain the location of the header files to include, and the libraries to be linked:

```
python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'
python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'
```
Include the compile flags and link flags in the makefile. Note, you need to include 2 shared libraries: `libtensorflow_framework.so.2` and `libtensorflow_cc.so.2`. (During compilation if some headers are not found you may have to include the directories containing those files specifically.)

Additionally you would have to include the following link flags before the shared libraries:
```
-Wl,--allow-multiple-definition -Wl,--whole-archive -l:libtensorflow_framework.so.2 -Wl,--whole-archive -Wl,--no-as-needed -l:libtensorflow_cc.so.2.3.0
```
After compilation, export `LD_LIBRARY_PATH` to point to the location of the shared libraries.
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path to .so files>
```
To run the binary, move to a location that contains PIXELAV events and `pix_2t.proc`. Then perform:
```
./bin/gen_xy_template
./bin/nn_gen_zp_template
```

