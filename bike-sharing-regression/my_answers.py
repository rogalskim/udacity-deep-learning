import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_dx(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.__set_node_counts(input_nodes, hidden_nodes, output_nodes)
        self.__init_weights()
        self.lr = learning_rate
        self.activation_function = sigmoid
        self.activation_function_dx = sigmoid_dx

    def __set_node_counts(self, input: int, hidden: int, output: int) -> None:
        self.input_nodes = input
        self.hidden_nodes = hidden
        self.output_nodes = output

    def __init_weights(self) -> None:
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5,
                                                        (self.input_nodes, self.hidden_nodes))
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                         (self.hidden_nodes, self.output_nodes))

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        for input_data, label in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(input_data)
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs,
                                                                        input_data, label,
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def __forward(self, input_data: np.ndarray) -> (np.ndarray, np.ndarray):
        hidden_inputs = np.matmul(input_data, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs

        return final_outputs, hidden_outputs

    def forward_pass_train(self, input_data: np.ndarray) -> (np.ndarray, np.ndarray):
        return self.__forward(input_data)

    def backpropagation(self,
                        final_outputs: np.ndarray,
                        hidden_outputs: np.ndarray,
                        input_data: np.ndarray,
                        label: np.float,
                        delta_weights_i_h: np.ndarray,
                        delta_weights_h_o: np.ndarray) -> (np.ndarray, np.ndarray):

        def output_activation_dx(x: np.ndarray) -> int:
            return 1

        error = label - final_outputs.item()
        output_error_term = output_activation_dx(hidden_outputs) * error

        hidden_error = self.weights_hidden_to_output * output_error_term
        hidden_input = np.matmul(input_data, self.weights_input_to_hidden)
        hidden_error_term = self.activation_function_dx(hidden_input) * hidden_error.T

        delta_weights_i_h += np.matmul(input_data[:, None], hidden_error_term)
        delta_weights_h_o += hidden_outputs[:, None] * output_error_term

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h: np.ndarray, delta_weights_h_o: np.ndarray, n_records: int) -> None:
        self.weights_hidden_to_output += delta_weights_h_o / n_records * self.lr
        self.weights_input_to_hidden += delta_weights_i_h / n_records * self.lr

    def run(self, features: np.ndarray) -> np.ndarray:
        return self.__forward(features)[0]


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 6000
learning_rate = 0.7
hidden_nodes = 20
output_nodes = 1
