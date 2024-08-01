package heml

// SimpleOptimizer implements a basic gradient descent optimizer
type SimpleOptimizer struct{}

// Update updates the weights and biases
func (s *SimpleOptimizer) Update(weights [][]float64, biases []float64, gradients [][]float64, lr float64) {
	for i := range weights {
		for j := range weights[i] {
			weights[i][j] -= lr * gradients[i][j]
		}
	}
	for i := range biases {
		biases[i] -= lr * gradients[0][i] // Simplified for demonstration purposes
	}
}
