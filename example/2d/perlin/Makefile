#################################################
#                                               #
#  C++ Makefile                                 #
#  Made by Phantomas <phantomas@phantomas.xyz>  #
#                                               #
#################################################

CXX = g++
ECHO = echo -e 
MKDIR = mkdir -p 
RM = rm -rf
SHELL = /bin/sh

CPPFLAGS += -I ../../../srcs

CXXFLAGS += -std=c++17
CXXFLAGS += $(DEBUG) -ggdb

LDFLAGS += -Wl,-E 
LDLIBS += -L ../../../bin -lworld_gen -lOpenCL -ltbb

OUTDIR = ../../../bin
REPORTDIR = reports

DEFAULT = "\033[00m"
GREEN = "\033[0;32m"
TEAL = "\033[0;36m"
RED = "\033[0;31m"

NAME = simple_perlin_noise
NAME = normalized_perlin_noise

#SRCS += ./simple.cpp
SRCS += ./normalized.cpp
SRCS += ../../utils/PgmImage.cpp

OBJS = $(SRCS:.cpp=.o)

all: $(NAME)

$(NAME): $(OBJS)
	@-$(MKDIR) $(OUTDIR)
	-$(CXX) -o $(OUTDIR)/$(NAME) $^ $(LDFLAGS) $(LDLIBS)\
	 && \
	 $(ECHO) $(GREEN) "[OK]" $(TEAL) $@ $(DEFAULT) || \
	 $(ECHO) $(RED) "[XX]" $(TEAL) $@ $(DEFAULT)

%.o: %.cpp
	-$(CXX) -c $< -o $@ $(CPPFLAGS) $(CXXFLAGS) \
	 && \
	 $(ECHO) $(GREEN) "[OK]" $(TEAL) $< $(DEFAULT) || \
	 $(ECHO) $(RED) "[XX]" $(TEAL) $< $(DEFAULT)

clean: 
	@-$(RM) $(OBJS)
	@-$(ECHO) $(TEAL) "Removing objects files" $(DEFAULT)

distclean: clean 
	@-$(RM) $(OUTDIR)/$(NAME)
	@-$(ECHO) $(TEAL) "Removing binary" $(DEFAULT)

re: distclean all

.PHONY: all clean cleanlog cleandep test_clean test_cleanlog test_cleandep distclean re tags

.SUFFIXES:
.SUFFIXES: .cpp .o
