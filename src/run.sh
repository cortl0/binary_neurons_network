#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

cpu_path="build/examples/minimal"
gpu_cuda_path="gpu/cuda"
message_prefix="=============== run.sh: "
message_postfix=" ==============="

echo ${message_prefix}begin${message_postfix}

echo ${message_prefix}run minimal bnn gpu cuda${message_postfix}
cd gpu/cuda
./run.sh
cd ../..
echo ${message_prefix}fin minimal bnn gpu cuda${message_postfix}

rm -r build
mkdir build
cd build
echo ${message_prefix}begin cmake${message_postfix}
cmake ..
echo ${message_prefix}end cmake${message_postfix}

echo ${message_prefix}begin make${message_postfix}
make
echo ${message_prefix}end make${message_postfix}
cd ..

echo ${message_prefix}run minimal bnn cpu${message_postfix}
./${cpu_path}/minimal
echo ${message_prefix}fin minimal bnn cpu${message_postfix}

echo ${message_prefix}end${message_postfix}
