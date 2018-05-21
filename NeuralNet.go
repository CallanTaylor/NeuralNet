package main

import "fmt"
import "math"
import "math/rand"
import "time"


type neuron struct {

  layer string
  weights []float64
  bais float64

}

func newNeuron(s string) neuron {

  rand.Seed(time.Now().UTC().UnixNano())
  weights := []float64{rand.Float64(), rand.Float64(), rand.Float64(), rand.Float64()}
  bais := rand.Float64()
  node := neuron{weights:weights, bais:bais, layer:s}
  return node

}


func predict(values [5]float64, nodes []neuron) float64 {

  answer := values[len(values) - 1]
  hiddenLayer := [4]float64{}
  pred := 0.0
  for i := 0; i < len(nodes); i++ {
    z := nodes[i].bais
    if nodes[i].layer == "H1" {
      for j := 0; j < len(nodes[i].weights); j++ {
        z += nodes[i].weights[j] * values[j]
      }
      hiddenLayer[i] = z
    } else {
      pred = nodes[i].bais
      for k := 0; k < len(hiddenLayer); k++ {
        pred += hiddenLayer[k] * nodes[i].weights[k]
      }
    }
  }
  pred = sigmoid(pred)
  error := math.Abs(answer - pred)
  fmt.Printf("Error: %.4f pred: %.4f target: %.1f\n", error, pred, answer)
  return error
}


func sigmoid(x float64) float64 {
  return 1 / (1 + math.Exp(-x))
}


func forwardPass(trainingData [5]float64, nodes []neuron) (float64, [4]float64) {

  pred := 0.0
  hiddenLayer := [4]float64{}
  for i := 0; i < len(nodes); i++ {
    z := nodes[i].bais
    if nodes[i].layer == "H1" {
      for j := 0; j < len(nodes[i].weights); j++ {
        z += nodes[i].weights[j] * trainingData[j]
      }
      hiddenLayer[i] = z
    } else {
      pred = nodes[i].bais
      for k := 0; k < len(hiddenLayer); k++ {
        pred += hiddenLayer[k] * nodes[i].weights[k]
      }
    }
  }
  return pred, hiddenLayer
}


func trainNN(trainingData [][5]float64, nodes []neuron) {
  iterations := 50000
  learningRate := 0.1

  for i := 0; i < iterations; i++ {

    /*Randomly choose a piece of Training data */
    trainingDataIndex := rand.Intn(len(trainingData))
    trainingPoints := trainingData[trainingDataIndex]

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
        if nodes[l].layer == "final" {

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
        if nodes[l].layer == "H1" {

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

func main() {

  predictionData := getPredictionData()
  trainingData := getTrainingData()
  nn := []neuron{newNeuron("H1"), newNeuron("H1"), newNeuron("H1"), newNeuron("H1"), newNeuron("final")}
  trainNN(trainingData, nn)

  totalError := 0.0

  for i := 0; i < len(predictionData); i++ {
    totalError += predict(predictionData[i], nn)
  }
  fmt.Println("Average Error:", totalError / 10.0)
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
