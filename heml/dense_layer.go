package heml

import "math/rand"

// DenseLayer represents a fully connected layer
type DenseLayer struct {
	inputSize  int
	outputSize int
	weights    [][]float64
	biases     []float64
	gradients  [][]float64
}

// NewDenseLayer creates a new dense layer with given input and output sizes
func NewDenseLayer(inputSize, outputSize int) *DenseLayer {
	// Initialize weights and biases with random values (simplified here)
	weights := make([][]float64, inputSize)
	for i := range weights {
		weights[i] = make([]float64, outputSize)
		for j := range weights[i] {
			weights[i][j] = rand.Float64() - 0.5 // Random initialization
		}
	}
	biases := make([]float64, outputSize)
	gradients := make([][]float64, inputSize)
	for i := range gradients {
		gradients[i] = make([]float64, outputSize)
	}
	return &DenseLayer{inputSize, outputSize, weights, biases, gradients}
}

// Forward pass through the dense layer
func (layer *DenseLayer) Forward(input []float64) []float64 {
	output := make([]float64, layer.outputSize)
	for i := 0; i < layer.outputSize; i++ {
		for j := 0; j < layer.inputSize; j++ {
			output[i] += input[j] * layer.weights[j][i]
		}
		output[i] += layer.biases[i]
	}
	return output
}

// Backward pass through the dense layer
func (layer *DenseLayer) Backward(input, gradOutput []float64) []float64 {
	gradInput := make([]float64, layer.inputSize)
	for i := 0; i < layer.inputSize; i++ {
		for j := 0; j < layer.outputSize; j++ {
			gradInput[i] += gradOutput[j] * layer.weights[i][j]
			layer.gradients[i][j] += input[i] * gradOutput[j]
		}
	}
	for i := 0; i < layer.outputSize; i++ {
		layer.gradients[0][i] += gradOutput[i]
	}
	return gradInput
}
