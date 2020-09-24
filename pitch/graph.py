from pydot import graph_from_dot_data
from chainer import Variable
from chainer.computational_graph import build_computational_graph
from network import Network
import numpy as np

net = Network()
net.to_gpu()
data = np.zeros((1, 3, 512, 512), dtype="float32")
variable = Variable(data)
variable.to_gpu()
outputs = net(variable, 3)

d = build_computational_graph(outputs).dump()
g = graph_from_dot_data(d)[0]
g.write_pdf("graph.pdf")
