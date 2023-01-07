package main

import (
	"fmt"
	"log"

	"github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	// Load the TensorFlow model
	model, err := tensorflow.LoadSavedModel("model", []string{"serve"}, nil)
	if err != nil {
		log.Fatal(err)
	}

	// Create a prediction session and make a prediction
	predictionSession, err := model.Session(nil)
	if err != nil {
		log.Fatal(err)
	}
	defer predictionSession.Close()

	// Pass in the input tensor
	inputTensor, err := tensorflow.NewTensor([][]float32{{3, 5, 7, 8, 9}})
	if err != nil {
		log.Fatal(err)
	}
	output, err := predictionSession.Run(
		map[tensorflow.Output]*tensorflow.Tensor{
			model.Operation("input_tensor").Output(0): inputTensor,
		},
		[]tensorflow.Output{
			model.Operation("output_tensor").Output(0),
		},
		nil,
	)
	if err != nil {
		log.Fatal(err)
	}

	// Print the prediction
	fmt.Println(output[0].Value())
}
