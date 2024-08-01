package heml

import (
	"fmt"
	"math"
	"sync"
)

// Model represents a neural network model
type Model struct {
	layers       []Layer
	optimizer    *SimpleOptimizer
	learningRate float64
	clipValue    float64
}

// NewModel creates a new empty model
func NewModel(learningRate, clipValue float64) *Model {
	return &Model{
		layers:       []Layer{},
		optimizer:    &SimpleOptimizer{},
		learningRate: learningRate,
		clipValue:    clipValue,
	}
}

// AddLayer adds a new layer to the model
func (model *Model) AddLayer(layer Layer) {
	model.layers = append(model.layers, layer)
}

// Predict makes a prediction using the model
func (model *Model) Predict(input []float64) []float64 {
	output := input
	for _, layer := range model.layers {
		output = layer.Forward(output)
	}
	return output
}

// Train trains the model on the given data
func (model *Model) Train(inputs [][]float64, labels [][]float64, epochs int, batchSize int) {
	var mu sync.Mutex
	var wg sync.WaitGroup

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		numBatches := (len(inputs) + batchSize - 1) / batchSize

		for i := 0; i < len(inputs); i += batchSize {
			batchInputs := inputs[i:min(i+batchSize, len(inputs))]
			batchLabels := labels[i:min(i+batchSize, len(labels))]

			wg.Add(1)
			go func(batchInputs, batchLabels [][]float64, batchIndex int) {
				defer wg.Done()

				batchLoss := 0.0

				// Initialize gradients to zero
				for _, layer := range model.layers {
					if denseLayer, ok := layer.(*DenseLayer); ok {
						for i := range denseLayer.gradients {
							for j := range denseLayer.gradients[i] {
								denseLayer.gradients[i][j] = 0
							}
						}
					}
				}

				for j, input := range batchInputs {
					// Forward pass
					output := model.Predict(input)
					loss := L1Loss(output, batchLabels[j])
					if math.IsNaN(loss) || math.IsInf(loss, 0) {
						fmt.Println("NaN or Inf detected in loss. Exiting.")
						return
					}
					batchLoss += loss

					// Backward pass
					gradOutput := make([]float64, len(output))
					for k := range output {
						gradOutput[k] = output[k] - batchLabels[j][k]
					}
					for k := len(model.layers) - 1; k >= 0; k-- {
						if denseLayer, ok := model.layers[k].(*DenseLayer); ok {
							gradOutput = denseLayer.Backward(input, gradOutput)
						}
					}
				}

				batchLoss /= float64(len(batchInputs))

				// Clip gradients before updating weights
				for _, layer := range model.layers {
					if denseLayer, ok := layer.(*DenseLayer); ok {
						model.clipGradients(denseLayer.gradients)

						mu.Lock()
						model.optimizer.Update(denseLayer.weights, denseLayer.biases, denseLayer.gradients, model.learningRate)
						mu.Unlock()
					}
				}

				mu.Lock()
				totalLoss += batchLoss
				fmt.Printf("\rEpoch: %d, Batch: %d/%d, Batch Loss: %f", epoch, batchIndex+1, numBatches, batchLoss)
				mu.Unlock()
			}(batchInputs, batchLabels, i/batchSize)
		}

		wg.Wait()
		fmt.Printf("\rEpoch: %d, Average Loss: %f\n", epoch, totalLoss/float64(numBatches))
	}
}

func (model *Model) clipGradients(gradients [][]float64) {
	for i := range gradients {
		for j := range gradients[i] {
			if gradients[i][j] > model.clipValue {
				gradients[i][j] = model.clipValue
			} else if gradients[i][j] < -model.clipValue {
				gradients[i][j] = -model.clipValue
			}
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
