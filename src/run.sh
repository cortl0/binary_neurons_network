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
echo "run cpu minimal";
./build/examples/minimal/cpu/minimal-cpu;
echo "run cuda minimal";
./build/examples/minimal/cuda/minimal-cuda;
echo "end";
