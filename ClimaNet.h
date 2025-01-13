#pragma once
#include <vector>
#include <torch/torch.h>

/*
 @breif: A neural network model definition using PyTorch's C++ API
*/ 

/*
	A neural network model definition using PyTorch's C++ API for clima modeling
	@input_size: The number of input features
	@hidden_size: The number of neurons in the first hidden layer
	@hidden_size2: The number of neurons in the second hidden layer
	@hidden_size3: The number of neurons in the third hidden layer
	@output_size: The number of output features
*/
struct ClimaNetImpl : torch::nn::Module {
	// Constructor for the neural network
	// Initializes the layers of the network and registers them as submodules
	ClimaNetImpl(int64_t input_size, int64_t hidden_size, int64_t hidden_size2, int64_t hidden_size3, int64_t output_size)
		: i2h(register_module("i2h", torch::nn::Linear(input_size, hidden_size))),         // Input to first hidden layer
		h2h(register_module("h2h", torch::nn::Linear(hidden_size, hidden_size2))),       // First to second hidden layer
		h2h2(register_module("h2h2", torch::nn::Linear(hidden_size2, hidden_size3))),    // Second to third hidden layer
		h2h3(register_module("h2h3", torch::nn::Linear(hidden_size3, hidden_size3))),    // Third hidden layer self-loop
		h2o(register_module("h2o", torch::nn::Linear(hidden_size3, output_size))) {
	}     // Final hidden layer to output layer

// The forward function defines how data flows through the network
	torch::Tensor forward(torch::Tensor x1) {
		// Pass input through the network with ReLU activations between layers
		torch::Tensor xm = torch::relu(i2h->forward(x1));           // Input to first hidden layer
		torch::Tensor xm2 = torch::relu(h2h->forward(xm));          // First to second hidden layer
		torch::Tensor xm3 = torch::relu(h2h2->forward(xm2));        // Second to third hidden layer
		torch::Tensor xo = torch::relu(h2h3->forward(xm3));         // Third hidden layer self-loop
		torch::Tensor output = h2o->forward(xo);                    // Final layer produces output

		return output; // Return the final output tensor
	}

	// Layer definitions
	torch::nn::Linear i2h, h2h, h2h2, h2h3, h2o;
};

// A convenience macro to define a shared pointer for the network
// This allows easy usage and management of the model in PyTorch workflows.
TORCH_MODULE(ClimaNet);
