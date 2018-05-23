package main

import "fmt"
import "math"
import "math/rand"
import "time"
import "flag"

var defaultSize int = 4
var hiddenLayerHelp string = "How many Neurons in the hidden layer"
var defaultIterations int = 50000

type neuron struct {

  layer int
  weights []float64
  bais float64

}

func newNeuron(l int, numOfWeights int) neuron {

  weights := []float64{}
  for i := 0; i < numOfWeights; i++ {
    rand.Seed(time.Now().UTC().UnixNano())
    weights = append(weights, rand.Float64() - 0.5)
  }
  rand.Seed(time.Now().UTC().UnixNano())
  bais := rand.Float64() - 0.5
  node := neuron{weights:weights, bais:bais, layer:l}
  return node

}


func predict(values []float64, nodes []neuron) float64 {

  error := 0.0
  if len(values) > 0 {
    answer := values[len(values) - 1]
    hiddenLayer := []float64{}
    pred := 0.0
    for i := 0; i < len(nodes); i++ {
      z := nodes[i].bais
      if nodes[i].layer == 1 {
        for j := 0; j < len(nodes[i].weights); j++ {
          z += nodes[i].weights[j] * values[j]
        }
        hiddenLayer = append(hiddenLayer, z)
      } else {
        pred = nodes[i].bais
        for k := 0; k < len(hiddenLayer); k++ {
          pred += hiddenLayer[k] * nodes[i].weights[k]
        }
      }
    }
    pred = sigmoid(pred)
    error = math.Abs(answer - pred)
    fmt.Printf("Error: %.4f pred: %.4f target: %.1f\n", error, pred, answer)
  }
  return error
}


func sigmoid(x float64) float64 {
  return 1 / (1 + math.Exp(-x))
}


func forwardPass(trainingData []float64, nodes []neuron) (float64, []float64) {

  pred := 0.0
  hiddenLayer := []float64{}
  for i := 0; i < len(nodes); i++ {
    z := nodes[i].bais
    if nodes[i].layer == 1 {
      for j := 0; j < len(trainingData) - 1; j++ {
        z += nodes[i].weights[j] * trainingData[j]
      }
      hiddenLayer = append(hiddenLayer, z)
    } else {
      pred = nodes[i].bais
      for k := 0; k < len(hiddenLayer); k++ {
        pred += hiddenLayer[k] * nodes[i].weights[k]
      }
    }
  }
  return pred, hiddenLayer
}


func trainNN(trainingData [][]float64, nodes []neuron, iterations int) {
  learningRate := 0.1

  for i := 0; i < iterations; i++ {

    /*Randomly choose a piece of Training data */
    trainingDataIndex := rand.Intn(len(trainingData))
    trainingPoints := trainingData[trainingDataIndex]

    if len(trainingPoints) > 0 {

      /*The target (y) = the last index of trainingData */
      target := trainingPoints[len(trainingPoints) - 1]

      /*Compute a prediction using that data then squish it to between 0 and 1 */
      z, a := forwardPass(trainingPoints, nodes)
      pred := sigmoid(z)

      /*The cost now equals (y - pred)^2 */

       for l := 0; l < len(nodes); l++ {

          /*
           * For any neurons in the final later we can compute the change in the value
           * of its weights using f(1). The partial derivitive of the cost with respect
           * to that weight.
           *  f(1)
           *  dCo     dZl   dAl   dCo
           *  ---  =  --- x --- x ---
           *  dWl     dWl   dZl   dAl
           *
           */
          if nodes[l].layer == 0 {

            dcost_dpred := 2 * (pred - target)

            dpred_dz := sigmoid(z) * (1 - sigmoid(z))

            for n := 0; n < len(nodes[l].weights); n++ {
              /*
               * A(l-1) = dz_dw[n] so no need to calculate dz_dw[n] which is the
               * corresponding value spit out by the hiddenLayer
               */
              nodes[l].weights[n] -= learningRate * (dcost_dpred * dpred_dz * a[n])
            }
            nodes[l].bais -= learningRate * (dcost_dpred * dpred_dz)
          }

          /*
           * For neuron in (l-1) we compute the change in theirs weights using f(2)
           * the partial derivitive of the cost with respect to that weight
           *     dCo      dZ(l-1)     dzl    dAl   dCo
           *  -------  =  ------- x ------ x --- x ---
           *  dW(l-1)     dW(l-1)   dA(l-1)  dZl   dAl
           */
          if nodes[l].layer == 1 {

          dcost_dpred := 2 * (pred - target)
          dpred_dz := sigmoid(z) * (1 - sigmoid(z))
          /* dz/dA(l-1) comes out to be the weight */
          dz_dAh := nodes[len(nodes) - 1].weights[l]

          for a := 0; a < len(nodes[l].weights); a++ {
            nodes[l].weights[a] -= learningRate * (dcost_dpred * dpred_dz * dz_dAh * trainingPoints[a])
          }
          nodes[l].bais -= learningRate * (dcost_dpred * dpred_dz * dz_dAh)
        }
      }
    }
  }
}

func main() {

  hiddenLayerSize := flag.Int("HiddenNeurons", defaultSize, hiddenLayerHelp)
  iterations := flag.Int("Iterations", defaultIterations, "iterations of training")
  flag.Parse()

  /* The size of inputs is the size of individual set of trainingData minus the
   * last value which represents the result
   */

  training := getData("WineClassification.txt")
  trainingData := prepareData(training)
  prediction := getData("WinePredictions.txt")
  predictionData := prepareData(prediction)
  inputSize := len(trainingData[0]) - 1

  nn := []neuron{}
  for j := 0; j < *hiddenLayerSize; j++ {
    nn = append(nn, newNeuron(1, inputSize))
  }
  nn = append(nn, newNeuron(0, *hiddenLayerSize))
  trainNN(trainingData, nn, *iterations)
  fmt.Println(len(nn))

  totalError := 0.0

  for i := 0; i < len(predictionData); i++ {
    totalError += predict(predictionData[i], nn)
  }
  fmt.Println("Average Error:", totalError / float64(len(predictionData)))
}

/*  Appendix to back propogation
 * ------------------------------------------------------------------------
 *  f(1)
 *  dCo     dZl   dAl   dCo
 *  ---  =  --- x --- x ---
 *  dWl     dWl   dZl   dAl
 *  change to Wl = trainingPoints[n] * dpred_dz * dcost_dpred
 *  (Where l equals last layer)
 *
 *  f(1a)
 *  dCo     dZl   dAl   dCo
 *  ---  =  --- x --- x ---
 *  dbl     dbl   dZl   dAl
 *  change to bl = 1 * dpred_dz * dcost_dpred
 *  (Where l equals last layer) (Because derivitive of x = 1)
 *
 *  -----------------------------------------------------------------------
 *
 *  For l - 1
 *    dCo       dZl      dAl   dCo
 *  ------  =  ------  x --- x ---
 *  dA(l-1)    dA(l-1)   dZl   dAl
 *  Where dzl / dA(l-1) turns out to be Wl
 *
 *  f(2)
 *  So
 *     dCo      dZ(l-1)     dzl    dAl   dCo
 *  -------  =  ------- x ------ x --- x ---
 *  dW(l-1)     dW(l-1)   dA(l-1)  dZl   dAl
 *  -----------------------------------------------------------------------
 */
