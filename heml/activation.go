package heml

import "math"

// ReLU activation function
func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// Softmax activation function
func Softmax(x []float64) []float64 {
	max := x[0]
	for _, v := range x {
		if v > max {
			max = v
		}
	}
	expSum := 0.0
	for i, v := range x {
		x[i] = math.Exp(v - max)
		expSum += x[i]
	}
	for i := range x {
		x[i] /= expSum
	}
	return x
}
