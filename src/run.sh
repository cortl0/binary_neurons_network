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
make -j6;
cd ..;
echo "run minimal-cpu";
./build/examples/minimal/cpu/minimal-cpu;
echo $?;
echo "run minimal-cuda";
./build/examples/minimal/cuda/minimal-cuda;
echo $?;
echo "end";
