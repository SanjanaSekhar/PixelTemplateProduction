root_flags= `root-config --cflags --ldflags --libs --glibs`
compiler_flags=-std=c++11 -Wall -g
vdt_dir_local=/home/sanjana/PixelCPE/include
vdt_dir_lpc=/uscms/home/ssekhar/nobackup/PixelCPE/include
tensorflow_dir=/home/sanjana/PixelCPE/tensorflow/
#eigen_dir = /home/sanjana/PixelCPE/tensorflow/bazel-tensorflow/external/eigen_archive
includes= -I. -I../cmssw_code/ -I$(vdt_dir_lpc) 
includes_TF = -I/home/sanjana/.virtualenvs/tf_dev/lib/python3.7/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=1 -I$(tensorflow_dir)bazel-tensorflow/  -L/home/sanjana/.virtualenvs/tf_dev/lib/python3.7/site-packages/tensorflow -Wl,--allow-multiple-definition -Wl,--whole-archive -l:libtensorflow_framework.so.2 -L$(tensorflow_dir)bazel-bin/tensorflow/ -Wl,--whole-archive -Wl,--no-as-needed -l:libtensorflow_cc.so.2.3.0  
DEPS= template_utils.h ranlux.c ../cmssw_code/* nn_reco.h 
EXE=  gen_zp_template compare_templates gen_zp_template2d gen_xy_template gen_xy_template2d test_template #nn_gen_zp_template
ODIR=../bin
BOOST_FLAG=-DSI_PIXEL_TEMPLATE_USE_BOOST

all: $(EXE)

nn_gen_zp_template: nn_gen_zp_template.cc $(DEPS)
	g++  nn_gen_zp_template.cc  -o $(ODIR)/nn_gen_zp_template $(compiler_flags)  $(root_flags)  $(includes) $(includes_TF) $(BOOST_FLAG) 

gen_zp_template: gen_zp_template.cc $(DEPS)
	g++ gen_zp_template.cc -o $(ODIR)/gen_zp_template $(compiler_flags)  $(root_flags) $(includes) $(BOOST_FLAG)

gen_zp_template2d: gen_zp_template2d.cc $(DEPS)
	g++ gen_zp_template2d.cc -o $(ODIR)/gen_zp_template2d $(compiler_flags)  $(root_flags) $(includes) $(BOOST_FLAG)

gen_xy_template: gen_xy_template.cc $(DEPS)
	g++ gen_xy_template.cc   -o $(ODIR)/gen_xy_template $(compiler_flags) $(root_flags) $(includes)

gen_xy_template2d: gen_xy_template2d.cc $(DEPS)
	g++ gen_xy_template2d.cc  -o $(ODIR)/gen_xy_template2d $(compiler_flags) $(root_flags) $(includes)

compare_templates: compare_templates.cc $(DEPS)
	g++ compare_templates.cc -o $(ODIR)/compare_templates $(compiler_flags) $(root_flags) $(includes)

test_template: test_template.cc $(DEPS)
	g++ test_template.cc -o $(ODIR)/test_template $(compiler_flags) $(root_flags) $(includes) $(BOOST_FLAG)

load_template_test: load_template_test.cc $(DEPS)
	g++ load_template_test.cc -o $(ODIR)/load_template_test $(compiler_flags) $(root_flags) $(includes) $(BOOST_FLAG)

clean:
	rm -f $(ODIR)/*

.PHONY: all clean

