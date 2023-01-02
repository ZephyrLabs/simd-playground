CXX=g++
CXXFLAGS=-mavx

init:
	mkdir build
	cd build

vec_add: x86/avx/vector/avx_add.cpp
	$(CXX) $(CXXFLAGS) x86/avx/vector/avx_add.cpp -o build/a.out
	./build/a.out
	rm ./build/a.out

vec_sub: x86/avx/vector/avx_sub.cpp
	$(CXX) $(CXXFLAGS) x86/avx/vector/avx_sub.cpp -o build/a.out
	./build/a.out
	rm ./build/a.out

vec_mul: x86/avx/vector/avx_mul.cpp
	$(CXX) $(CXXFLAGS) x86/avx/vector/avx_mul.cpp -o build/a.out
	./build/a.out
	rm ./build/a.out

vec_div: x86/avx/vector/avx_div.cpp
	$(CXX) $(CXXFLAGS) x86/avx/vector/avx_div.cpp -o build/a.out
	./build/a.out
	rm ./build/a.out

tensor_add: x86/avx/tensor/avx_add.cpp
	$(CXX) $(CXXFLAGS) x86/avx/tensor/avx_add.cpp -o build/a.out
	./build/a.out
	rm ./build/a.out

tensor_sub: x86/avx/tensor/avx_sub.cpp
	$(CXX) $(CXXFLAGS) x86/avx/tensor/avx_sub.cpp -o build/a.out
	./build/a.out
	rm ./build/a.out

tensor_mul: x86/avx/tensor/avx_mul.cpp
	$(CXX) $(CXXFLAGS) x86/avx/tensor/avx_mul.cpp -o build/a.out
	./build/a.out
	rm ./build/a.out

clean:
	rm -rf /build