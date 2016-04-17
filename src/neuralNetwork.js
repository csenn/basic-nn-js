// import math from 'mathjs'
import _ from 'lodash';
import {
  addMatrix,
  subtractMatrix,
  multiplyMatrix,
  crossProduct,
  transposeMatrix,
  applyFunctionOverMatrix,
  zeroMatrix
} from './matrixUtils';

// http://stackoverflow.com/questions/25582882/javascript-math-random-normal-distribution-gaussian-bell-curve
function gaussian(mean, stdev) {
  let y2;
  let useLast = false;
  return function () {
    let y1;
    if (useLast) {
      y1 = y2;
      useLast = false;
    }
    else {
      let x1, x2, w;
      do {
        x1 = 2.0 * Math.random() - 1.0;
        x2 = 2.0 * Math.random() - 1.0;
        w = x1 * x1 + x2 * x2;
      } while (w >= 1.0);
      w = Math.sqrt((-2.0 * Math.log(w)) / w);
      y1 = x1 * w;
      y2 = x2 * w;
      useLast = true;
    }

    const retval = mean + stdev * y1;
    return retval;
    //    if(retval > 0)
    //        return retval;
    //    return -retval;
  };
}


function calculateSigmoid(num) {
  return 1 / (1 + Math.pow(Math.E, -num));
}

function calculateSigmoidPrime(num) {
  const sigmoid = calculateSigmoid(num)
  return sigmoid * (1 - sigmoid);
}

function calculateCostDerivative(output, y) {
  return subtractMatrix(output, y);
}

export function backProp(x, y, biases, weights) {

  const nablaB = zeroMatrix(biases);
  const nablaW = zeroMatrix(weights);

  let activation = x;
  const activations = [x];
  const zs = [];
  // Feedforward
  for (let i = 0; i < biases.length; i++) {
    const cross = crossProduct(weights[i], activation);
    const z = addMatrix(cross, biases[i]);
    zs.push(z);
    activation = applyFunctionOverMatrix(z, calculateSigmoid);
    activations.push(activation);
  }

  const error = calculateCostDerivative(activation, y);
  const derivativeZ = applyFunctionOverMatrix(zs[zs.length - 1], calculateSigmoidPrime);

  // This is the error in the last layer
  let delta = multiplyMatrix(error, derivativeZ);

  nablaB[nablaB.length - 1] = delta;
  const tranposeSecondLastLayer = transposeMatrix(activations[activations.length - 2]);
  nablaW[nablaW.length - 1] = crossProduct(delta, tranposeSecondLastLayer);

  for (let i = biases.length - 1; i > 0; i--) {
    const z = zs[i - 1];
    const layerDerivative = applyFunctionOverMatrix(z, calculateSigmoidPrime);
    delta = multiplyMatrix(
      layerDerivative,
      crossProduct(transposeMatrix(weights[i]), delta)
    );
    nablaB[i - 1] = delta;
    nablaW[i - 1] = crossProduct(delta, transposeMatrix(activations[i - 1]));
  }
  return { nablaB, nablaW };
}

export function updateMiniBatch(batch, eta, initBiases, initWeights) {
  // Initialize to zero values
  const nablaB = zeroMatrix(initBiases);
  const nablaW = zeroMatrix(initWeights);
  const biases = initBiases;
  const weights = initWeights;

  // Using a single mini-batch, adjust the gradient for each backprop
  for (let i = 0; i < batch.length; i++) {
    const deltas = backProp(batch[i].x, batch[i].y, biases, weights);
    for (let j = 0; j < nablaB.length; j++) {
      nablaB[j] = addMatrix(nablaB[j], deltas.nablaB[j]);
      nablaW[j] = addMatrix(nablaW[j], deltas.nablaW[j]);
    }
  }

  const adjustFunc = num => num * (eta / batch.length);
  for (let i = 0; i < biases.length; i++) {
    biases[i] = subtractMatrix(
      biases[i],
      applyFunctionOverMatrix(nablaB[i], adjustFunc)
    )
    weights[i] = subtractMatrix(
      weights[i],
      applyFunctionOverMatrix(nablaW[i], adjustFunc)
    );
  }
  return { biases, weights };
}

export function feedForward(a, biases, weights) {
  for (let i = 0; i < biases.length; i++) {
    a = applyFunctionOverMatrix(addMatrix(
      crossProduct(weights[i], a),
      biases[i]
    ), calculateSigmoid);
  }
  return a;
}

export function evaluateTestData(testData, biases, weights) {

  const testMap = {}
  let count = 0;

  for (let i = 0; i < testData.length; i++) {
    const xResult = _.flattenDeep(feedForward(testData[i].x, biases, weights));
    const expectedIndex = testData[i].yIndex;
    let max = 0;
    let actualIndex = 0;
    for (let j = 0; j < xResult.length; j++) {
      if (xResult[j] > max) {
        max = xResult[j];
        actualIndex = j;
      }
    }
    if (!testMap[expectedIndex]) {
      testMap[expectedIndex] = {
        correct: [],
        wrong: {}
      }
    }
    if (expectedIndex === actualIndex) {
      count++;
      testMap[expectedIndex].correct.push(i)
    } else {
      if (!testMap[expectedIndex].wrong[actualIndex]) {
        testMap[expectedIndex].wrong[actualIndex] = []
      }
      testMap[expectedIndex].wrong[actualIndex].push(i)
    }
  }

  return {
    testMap,
    count
  }
  return results;
}

// with miniBatchSize = 4
// in - [1,1,1,1,1,1,1,1,1,1]
// out - [[1,1,1,1], [1,1,1,1], [1,1]]
export function splitIntoMiniBatches(trainingData, miniBatchSize) {
  const miniBatches = [];
  for (let i = 0; i < trainingData.length; i++) {
    if (i % miniBatchSize === 0) {
      miniBatches.push([]);
    }
    miniBatches[miniBatches.length - 1].push(trainingData[i]);
  }
  return miniBatches;
}



export function printSnapshot(biases, weights, epoch, onStateUpdate, testData) {
  const snapshot = {
    biases: biases,
    weights: weights
  }
  if (testData) {
    const {testMap, count} = evaluateTestData(testData, biases, weights);
    snapshot.testResults = testMap;
    if (epoch > 0) {
      console.log(`Epoch ${epoch} complete: ${count} out of ${testData.length}`)
    }
  } else {
    if (epoch > 0) {
      console.log(`Epoch ${epoch} complete`);
    }
  }

  if (onStateUpdate) {
    onStateUpdate(snapshot, epoch)
  }
}

export function runEpochs(options, initialBiases, initialWeights) {
  const { trainingData, epochs, miniBatchSize, eta, testData, onStateUpdate } = options;

  console.log(`Training data points: ${trainingData.length}`)
  printSnapshot(initialBiases, initialWeights, 0, onStateUpdate, testData)

  for (let i = 0; i < epochs; i++) {
    const miniBatches = splitIntoMiniBatches(_.shuffle(trainingData), miniBatchSize);

    let curWeights = initialWeights;
    let curBiases = initialBiases;
    miniBatches.forEach((batch, index) => {
      const { weights, biases } = updateMiniBatch(batch, eta, curBiases, curWeights);
      curWeights = weights;
      curBiases = biases;
    });

    printSnapshot(curBiases, curWeights, i + 1, onStateUpdate, testData)
  }
}

const guassianGenerator = gaussian(0, 1);

export function generateBiases(sizes) {
  const biases = [];
  for (let i = 0; i < sizes.length - 1; i++) {
    biases[i] = [];
    for (let j = 0; j < sizes[i + 1]; j++) {
      biases[i][j] = [guassianGenerator()];
    }
  }
  return biases;
}

// This stores matrix values a little backwards, with rows being target nodes
// and columns being start nodes
export function generateWeights(sizes) {
  const weights = [];
  for (let i = 0; i < sizes.length - 1; i++) {
    weights[i] = [];
    for (let j = 0; j < sizes[i + 1]; j++) {
      weights[i][j] = [];
      for (let k = 0; k < sizes[i]; k++) {
        weights[i][j][k] = guassianGenerator();
      }
    }
  }
  return weights;
}

export function runNeuralNetwork(options) {
  console.log('Starting...');
  const biases = generateBiases(options.sizes);
  const weights = generateWeights(options.sizes);
  runEpochs(options, biases, weights);
}
