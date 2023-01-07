package predictive

import (
	"errors"

	"github.com/tensorflow/tensorflow/tensorflow/go"
)

// NeuralNetwork represents a neural network model for predicting data.
type NeuralNetwork struct {
	model *tensorflow.SavedModel
}

// NewNeuralNetwork creates a new neural network model for prediction.
func NewNeuralNetwork(modelPath string) (*NeuralNetwork, error) {
	model, err := tensorflow.LoadSavedModel(modelPath)
	if err != nil {
		return nil, err
	}

	return &NeuralNetwork{
		model: model,
	}, nil
}

// Predict runs the neural network model on the given input data and returns the
// predicted output.
func (nn *NeuralNetwork) Predict(input *tensorflow.Tensor) (*tensorflow.Tensor, error) {
	if nn.model == nil {
		return nil, errors.New("neural network model not initialized")
	}

	output, err := nn.model.Session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{
			nn.model.Graph.Operation("input").Output(0): input,
		},
		[]tensorflow.Output{
			nn.model.Graph.Operation("output").Output(0),
		},
		nil,
	)
	if err != nil {
		return nil, err
	}

	return output[0], nil
}
