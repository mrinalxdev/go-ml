package main

import (
	"fmt"
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
    yRows, yCols := y.Dims()
    
    fmt.Printf("X dimensions: %d x %d\n", nSamples, nFeatures)
    fmt.Printf("y dimensions: %d x %d\n", yRows, yCols)
    
    if yRows != nSamples || yCols != 1 {
        panic(fmt.Sprintf("Incompatible dimensions. Expected y to be %dx1, but got %dx%d", nSamples, yRows, yCols))
    }

    lr.Weights = mat.NewVecDense(nFeatures, nil)

    yVec := y.ColView(0)

    for epoch := 0; epoch < epochs; epoch++ {
        predictions := mat.NewVecDense(nSamples, nil)
        predictions.MulVec(X, lr.Weights)
        for i := 0; i < nSamples; i++ {
            predictions.SetVec(i, predictions.AtVec(i) + lr.Bias)
        }

        errors := mat.NewVecDense(nSamples, nil)
        errors.SubVec(yVec, predictions)

        gradients := mat.NewVecDense(nFeatures, nil)
        gradients.MulVec(X.T(), errors)
        gradients.ScaleVec(-2.0/float64(nSamples), gradients)

        lr.Weights.AddScaledVec(lr.Weights, -learningRate, gradients)
        lr.Bias -= learningRate * mat.Sum(errors) * -2.0 / float64(nSamples)
    }
}

func (lr *LinearRegression) Predict(X *mat.Dense) *mat.VecDense {
    nSamples, _ := X.Dims()
    predictions := mat.NewVecDense(nSamples, nil)
    predictions.MulVec(X, lr.Weights)
    biasVec := mat.NewVecDense(nSamples, nil)
    for i := 0; i < nSamples; i++ {
        biasVec.SetVec(i, lr.Bias)
    }
    predictions.AddVec(predictions, biasVec)
    return predictions
}

func NormalizeData(X *mat.Dense) (*mat.Dense, []float64, []float64) {
	nSamples, nFeatures := X.Dims()
	normalizedX := mat.NewDense(nSamples, nFeatures, nil)
	means := make([]float64, nFeatures)
	stdDevs := make([]float64, nFeatures)

	for j := 0; j < nFeatures; j++ {
		col := X.ColView(j)
		means[j] = stat.Mean(mat.Col(nil, j, X), nil)
		stdDevs[j] = stat.StdDev(mat.Col(nil, j, X), nil)

		for i := 0; i < nSamples; i++ {
			normalizedX.Set(i, j, (col.AtVec(i)-means[j])/stdDevs[j])
		}
	}

	return normalizedX, means, stdDevs
}

func CalculateMetrics(y, predictions *mat.VecDense) (mse, mae, r2 float64) {
	nSamples := y.Len()

	diff := mat.NewVecDense(nSamples, nil)
	diff.SubVec(y, predictions)
	diff.MulElemVec(diff, diff)
	mse = mat.Sum(diff) / float64(nSamples)

	absDiff := mat.NewVecDense(nSamples, nil)
	absDiff.SubVec(y, predictions)
	for i := 0; i < nSamples; i++ {
		absDiff.SetVec(i, math.Abs(absDiff.AtVec(i)))
	}
	mae = mat.Sum(absDiff) / float64(nSamples)

	yMean := stat.Mean(mat.Col(nil, 0, y), nil)
	ssTotal, ssResidual := 0.0, 0.0
	for i := 0; i< nSamples; i++ {
		ssTotal += math.Pow(y.AtVec(i) -yMean, 2)
		ssResidual += math.Pow(y.AtVec(i) - predictions.AtVec(i), 2)
	}
	r2 = 1 - (ssResidual / ssTotal)
	
	return
}