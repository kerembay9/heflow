package heml

// Layer interface defines the basic structure of a layer in the neural network
type Layer interface {
	Forward(input []float64) []float64
}
