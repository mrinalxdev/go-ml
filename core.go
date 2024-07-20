package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type LinearRegression struct {
	Weights *mat.VecDense
	Bias    float64
}

func NewLinearRegression() *LinearRegression {
	return &LinearRegression{
		Weights: mat.NewVecDense(1, nil),
		Bias:    0,
	}
}

func (lr *LinearRegression) Train(X, y *mat.Dense, learningRate float64, epochs int) {
	nSamples, nFeatures := X.Dims()
	lr.Weights = mat.NewVecDense(nFeatures, nil)

	for epoch := 0; epoch < epochs; epoch++ {
		predictions := mat.NewVecDense(nSamples, nil)
		predictions.MulVec(X, lr.Weights)
		predictions.AddVec(predictions, mat.NewVecDense(nSamples, []float64{lr.Bias}))

		errors := mat.NewVecDense(nSamples, nil)
		errors.SubVec(y.ColView(0), predictions)
		gradients := mat.NewVecDense(nFeatures, nil)
		gradients.MulVec(X.T(), errors)
		gradients.ScaleVec(-2.0/float64(nSamples), gradients)
		
	}
}
