package main

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/opts"
	"github.com/olekukonko/tablewriter"
	"gonum.org/v1/gonum/mat"
)

func loadCSV(filename string) (*mat.Dense, *mat.Dense, []string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, nil, nil, err
	}
	features := records[0][:len(records[0])-1]
	nSamples := len(records) - 1
	nFeatures := len(features)

	X := mat.NewDense(nSamples, nFeatures, nil)
	Y := mat.NewDense(nSamples, 1, nil)

	for i, record := range records[1:] {
		for j, value := range record[:nFeatures] {
			floatValue, err := strconv.ParseFloat(value, 64)
			if err != nil {
				return nil, nil, nil, err
			}
			X.Set(i, j, floatValue)
		}
		yValue, err := strconv.ParseFloat(record[nFeatures], 64)
		if err != nil {
			return nil, nil, nil, err
		}
		Y.Set(i, 0, yValue)
	}

	return X, Y, features, nil
}

func plotResults(_, Y, predictions *mat.Dense, filename string) {
	scatter := charts.NewScatter()
	scatter.SetGlobalOptions(
		charts.WithTitleOpts(opts.Title{Title: "Actual vs Predicted"}),
		charts.WithXAxisOpts(opts.XAxis{Name: "Actual"}),
		charts.WithYAxisOpts(opts.YAxis{Name: "Predicted"}),
	)

	var data []opts.ScatterData
	for i := 0; i < Y.RawMatrix().Rows; i++ {
		data = append(data, opts.ScatterData{
			Value : []float64{Y.At(i, 0), predictions.At(i, 0)},
			SymbolSize: 10,
			Name : fmt.Sprintf("Points %d", i),
		})
	}

	scatter.AddSeries("scatter", data)
	f, _ := os.Create(filename)
	scatter.Render(f)
}

func main() {
	fmt.Println("Welcome to the Machine Learning Tool!")
	fmt.Println("=====================================")

	var filename string
	fmt.Print("Enter the CSV file name (or press Enter for demo data): ")
	fmt.Scanln(&filename)

	var X, y *mat.Dense
	var features []string
	var err error

	if filename == "" {
		X, y = generateDemoData(100, 3)
		features = []string{"Feature 1", "Feature 2", "Feature 3"}
	} else {
		X, y, features, err = loadCSV(filename)
		if err != nil {
			fmt.Printf("Error loading CSV: %v\n", err)
			return
		}
	}

	fmt.Printf("X dimensions: %d x %d\n", X.RawMatrix().Rows, X.RawMatrix().Cols)
    fmt.Printf("y dimensions: %d x %d\n", y.RawMatrix().Rows, y.RawMatrix().Cols)

    normalizedX, means, stdDevs := NormalizeData(X)

    model := NewLinearRegression()
    model.Train(normalizedX, y, 0.01, 1000)

    predictions := model.Predict(normalizedX)
    
    yVec := mat.NewVecDense(y.RawMatrix().Rows, nil)
    yVec.ColViewOf(y, 0)

    mse, mae, r2 := CalculateMetrics(yVec, predictions)

	fmt.Println("\nModel Results:")
	fmt.Println("==============")
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Metric", "Value"})
	table.Append([]string{"Mean Squared Error", fmt.Sprintf("%.4f", mse)})
	table.Append([]string{"Mean Absolute Error", fmt.Sprintf("%.4f", mae)})
	table.Append([]string{"R-squared", fmt.Sprintf("%.4f", r2)})
	table.Render()

	fmt.Println("\nModel Coefficients:")
	fmt.Println("===================")
	table = tablewriter.NewWriter(os.Stdout)
	table.SetHeader(append([]string{"Metric"}, features...))
	coefficients := make([]string, len(features))
	for i := 0; i < len(features); i++ {
		coefficients[i] = fmt.Sprintf("%.4f", model.Weights.AtVec(i))
	}
	table.Append(append([]string{"Coefficient"}, coefficients...))
	table.Append(append([]string{"Mean"}, floatsToStrings(means)...))
	table.Append(append([]string{"Std Dev"}, floatsToStrings(stdDevs)...))
	table.Render()

	fmt.Printf("\nBias: %.4f\n", model.Bias)

	plotResults(X, y, mat.DenseCopyOf(predictions), "scatter_plot.html")
	fmt.Println("\nScatter plot saved as 'scatter_plot.html'")

	for {
		fmt.Print("\nEnter values for prediction (comma-separated) or 'q' to quit: ")
		var input string
		fmt.Scanln(&input)
		if strings.ToLower(input) == "q" {
			break
		}

		values := strings.Split(input, ",")
		if len(values) != len(features) {
			fmt.Printf("Error: Expected %d values, got %d\n", len(features), len(values))
			continue
		}

		testX := mat.NewDense(1, len(features), nil)
		for i, value := range values {
			floatValue, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
			if err != nil {
				fmt.Printf("Error parsing value: %v\n", err)
				continue
			}
			testX.Set(0, i, (floatValue-means[i])/stdDevs[i])
		}

		prediction := model.Predict(testX).AtVec(0)
		fmt.Printf("Predicted value: %.4f\n", prediction)
	}

	fmt.Println("Thank you for using the Machine Learning Tool!")
}

func generateDemoData(nSamples, nFeatures int) (*mat.Dense, *mat.Dense) {
	X := mat.NewDense(nSamples, nFeatures, nil)
	y := mat.NewDense(nSamples, 1, nil)

	for i := 0; i < nSamples; i++ {
		sum := 0.0
		for j := 0; j < nFeatures; j++ {
			x := rand.Float64()*10 - 5
			X.Set(i, j, x)
			sum += x
		}
		noise := rand.NormFloat64() * 0.5
		yVal := sum + noise
		y.Set(i, 0, yVal)
	}

	return X, y
}

func floatsToStrings(floats []float64) []string {
	strings := make([]string, len(floats))
	for i, f := range floats {
		strings[i] = fmt.Sprintf("%.4f", f)
	}
	return strings
}