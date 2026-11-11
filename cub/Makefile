
WIN = 0
WIN64 = 0

ifeq ($(WIN),1)
	CUDA_ROOT = $(shell echo $$CUDA_PATH)
	NVCC = "$(CUDA_ROOT)/bin/nvcc"
	EXT = dll
ifeq ($(WIN64),1)
	NVCCFLAGS += -m64
else
	NVCCFLAGS += -m32
endif
	NVCCFLAGS += -Xptxas -v -Xcudafe -\# -shared 
else
	NVCC = "$(shell which nvcc)"
	CUDA_ROOT = $(shell dirname $(NVCC))/../
	EXT = so
	NVCCFLAGS += -Xptxas -v -Xcudafe -\# -shared \
			-Xcompiler -fPIC -Xcompiler -fvisibility=hidden
endif

COMMA = ,
ifdef sm
	SM_ARCH = $(subst $(COMMA),-,$(sm))
else 
	SM_ARCH = 610
endif

ifeq (610, $(findstring 610, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_61,code=\"sm_61,compute_61\" 
    SM_DEF 		+= -DSM610
endif
ifeq (620, $(findstring 620, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_62,code=\"sm_62,compute_62\" 
    SM_DEF 		+= -DSM620
endif
ifeq (700, $(findstring 700, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_70,code=\"sm_70,compute_70\" 
    SM_DEF 		+= -DSM700
endif
ifeq (750, $(findstring 750, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_75,code=\"sm_75,compute_75\" 
    SM_DEF 		+= -DSM750
endif
ifeq (800, $(findstring 800, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_80,code=\"sm_80,compute_80\" 
    SM_DEF 		+= -DSM800
endif
ifeq (860, $(findstring 860, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_86,code=\"sm_86,compute_86\" 
    SM_DEF 		+= -DSM860
endif
ifeq (890, $(findstring 890, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_89,code=\"sm_89,compute_89\" 
    SM_DEF 		+= -DSM890
endif
ifeq (900, $(findstring 900, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_90,code=\"sm_90,compute_90\" 
    SM_DEF 		+= -DSM900
endif
ifeq (1000, $(findstring 1000, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_100,code=\"sm_100,compute_100\" 
    SM_DEF 		+= -DSM1000
endif
ifeq (1200, $(findstring 1200, $(SM_ARCH)))
    SM_TARGETS 	+= -gencode=arch=compute_120,code=\"sm_120,compute_120\" 
    SM_DEF 		+= -DSM1200
endif

DEPS = ./Makefile \
	sort_engine.cu \
	sort_engine.h

LIBNAME = sort_engine

all: $(LIBNAME).$(EXT)
	touch built

clean :
	rm -f  *.dll *.so *.lib *.exp built

$(LIBNAME).$(EXT) : $(DEPS)
	$(NVCC) $(SM_TARGETS) $(SM_DEF) -o $@ sort_engine.cu $(NVCCFLAGS) -O3  

