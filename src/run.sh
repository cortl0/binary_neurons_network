#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

rm -r build;
echo "build bnn libs";
mkdir build;
cd build;
cmake ..;
make;
cd ..;
echo "run bnn-cuda-app";
./build/gpu/cuda/app/bnn-cuda-app;
echo "run minimal";
./build/examples/minimal/minimal;
echo "success";
