package heml

// L1Loss computes the mean absolute error
func L1Loss(predicted, actual []float64) float64 {
	loss := 0.0
	for i := range actual {
		loss += abs(predicted[i] - actual[i])
		// fmt.Println("loss: ", loss)
		// fmt.Println("actual: ", actual[i], "predicted: ", predicted[i])
		//wait for 1 second
		// time.Sleep(time.Second)
	}
	return loss / float64(len(actual))
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
