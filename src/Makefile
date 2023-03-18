CXX=g++
CXX_FLAGS= -O3 -g -std=c++17
DEFS=-DNDEBUG

default: main.x

%.o: %.cpp
	${CXX} -c ${CXX_FLAGS} ${DEFS} $<

main.o: header.hpp
buildMatrix.o: header.hpp
display.o: header.hpp

MAIN_OBJ=main.o display.o buildMatrix.o

main.x: ${MAIN_OBJ}
	${CXX} ${CXX_FLAGS} ${DEFS} -o main.x $^

clean:
	-rm *.o *.x
