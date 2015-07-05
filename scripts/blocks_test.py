"""Blocks test"""

from theano import tensor;
from blocks.bricks import MLP;
from blocks.bricks import Rectifier, Softmax;
from blocks.initialization import IsotropicGaussian, Constant;
from fuel.datasets.hdf5 import H5PYDataset;
from fuel.streams import DataStream;
from fuel.schemes import SequentialScheme;
from fuel.transformers import Flatten;
from blocks.algorithms import GradientDescent, Scale;
from blocks.bricks.cost import CategoricalCrossEntropy;
from blocks.bricks.cost import MisclassificationRate;
from blocks.graph import ComputationGraph;

train_set=H5PYDataset("dataset.hdf5", which_sets=('train', ))
test_set=H5PYDataset("dataset.hdf5", which_sets=('test', ));

data_stream = Flatten(DataStream.default_stream(train_set,
                                                iteration_scheme=SequentialScheme(train_set.num_examples,
                                                                                  batch_size=5)));
                                                
data_stream_test = Flatten(DataStream.default_stream(test_set,
                                                     iteration_scheme=SequentialScheme(test_set.num_examples,
                                                                                       batch_size=2)));

x = tensor.matrix('features');
y = tensor.lmatrix('targets');

mlp = MLP(activations=[Rectifier(), Softmax()], dims=[1000, 800, 7],
          weights_init=IsotropicGaussian(), biases_init=Constant(0.1));

mlp.initialize();
                    
y_hat=mlp.apply(x);

cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat);
error = MisclassificationRate().apply(y.flatten(), y_hat)
cg = ComputationGraph(cost)

algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                            step_rule=Scale(learning_rate=0.001))

from blocks.extensions.monitoring import DataStreamMonitoring

monitor = DataStreamMonitoring(variables=[cost, error], data_stream=data_stream_test, prefix="test")

from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
main_loop = MainLoop(data_stream=data_stream, algorithm=algorithm,
                     extensions=[monitor, FinishAfter(after_n_epochs=100), Printing()])

main_loop.run()