#
#   Binary Neurons Network
#   created by Ilya Shishkin
#   cortl@8iter.ru
#   https://github.com/cortl0/binary_neurons_network
#   licensed by GPL v3.0
#

rm -r build-bnn_cpu
cmake -S cpu -B build-bnn_cpu
make -C build-bnn_cpu

rm -r examples/build-minimal
cmake -S examples/minimal -B examples/build-minimal
make -C examples/build-minimal

chmod u+x examples/build-minimal/minimal
./examples/build-minimal/minimal
