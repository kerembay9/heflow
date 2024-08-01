package main

import (
	"fmt"
	"log"

	"heflow/heml"

	"github.com/petar/GoMNIST"
)

func main() {
	// Load the MNIST training and test data
	train, _, err := GoMNIST.Load("./mnist/")
	if err != nil {
		log.Fatalf("Failed to load MNIST dataset: %v", err)
	}

	// Prepare training data
	trainInputs := make([][]float64, len(train.Images))
	trainLabels := make([][]float64, len(train.Labels))
	for i := range train.Images {
		trainInputs[i] = make([]float64, 784) // Flatten 28x28 image to 784-element vector
		for j := 0; j < 784; j++ {
			trainInputs[i][j] = float64(train.Images[i][j]) / 255.0 // Normalize pixel values
		}
		trainLabels[i] = make([]float64, 10) // One-hot encode labels
		trainLabels[i][train.Labels[i]] = 1.0
	}

	// Create a new model
	model := heml.NewModel(0.0000001, 1)

	// Add layers to the model
	model.AddLayer(heml.NewDenseLayer(784, 128)) // Input size 784 (28x28 pixels), output size 128
	model.AddLayer(heml.NewDenseLayer(128, 64))  // Input size 128, output size 64
	model.AddLayer(heml.NewDenseLayer(64, 10))   // Input size 64, output size 10 (10 classes)

	// Train the model with batch learning
	model.Train(trainInputs, trainLabels, 4, 32)

	// Make a prediction on a single test input (using the first training image for simplicity)
	output := model.Predict(trainInputs[0])
	fmt.Println("Prediction:", findMaxValueIndex(output))
	fmt.Println("Actual:", findMaxValueIndex(trainLabels[0]))
}

func findMaxValueIndex(arr []float64) int {
	maxIndex := 0
	maxValue := arr[0]
	for i, value := range arr {
		if value > maxValue {
			maxValue = value
			maxIndex = i
		}
	}
	return maxIndex
}

func findMaxValue(arr []float64) float64 {
	maxValue := arr[0]
	for _, value := range arr {
		if value > maxValue {
			maxValue = value
		}
	}
	return maxValue
}
