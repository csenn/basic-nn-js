import _ from 'lodash';
import { gaussian } from './statUtils';
import {
  addMatrix,
  subtractMatrix,
  multiplyMatrix,
  crossProduct,
  transposeMatrix,
  applyFunctionOverMatrix,
  zeroMatrix
} from './matrixUtils';

// Neuron Functions
//https://theclevermachine.wordpress.com/2014/09/08/derivation-derivatives-for-common-neural-network-activation-functions/
const ACTIVATION_TYPE = {
  sigmoid: {
    func: (num) => {
      return 1 / (1 + Math.pow(Math.E, -num));
    },
    derivative: (num) => {
      const output = ACTIVATION_TYPE.sigmoid.func(num);
      return output * (1 - output);
    }
  },
  tanh: {
    func: (num) => {
      return Math.tanh(num);
    },
    derivative: (num) => {
     const tanh = ACTIVATION_TYPE.tanh.func(num);
     return 1 - tanh * tanh;
    }
  },
  relu: {
    func: (num) => {
      if (num > 0) {
        return num;
      }
      return 0;
    },
    derivative: (num) => {
      if (num > 0) {
        return 1;
      }
      return 0;
    }
  }
}

// Cost functions
const COST = {
  quadratic: {
    delta: (a, y, z, activationType) => {
      const rateChangeOutput = applyFunctionOverMatrix(z, ACTIVATION_TYPE[activationType].derivative);
      const rateChangeCost = subtractMatrix(a, y);
      return multiplyMatrix(rateChangeCost, rateChangeOutput);
    }
  },
  // Notice how this does not need z or activationType, it's derivative is the primary
  // value of cross entropy function
  crossEntropy: {
    delta: (a, y) => subtractMatrix(a, y)
  }
};


export function backProp(x, y, biases, weights, costFunction, activationType) {

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
    activation = applyFunctionOverMatrix(z, ACTIVATION_TYPE[activationType].func);
    activations.push(activation);
  }

  // Get Delta in final layer
  let delta = COST[costFunction].delta(activation, y, zs[zs.length - 1], activationType);

  nablaB[nablaB.length - 1] = delta;
  nablaW[nablaW.length - 1] = crossProduct(
    delta,
    transposeMatrix(activations[activations.length - 2])
  );

  for (let i = biases.length - 1; i > 0; i--) {
    const z = zs[i - 1];
    const layerDerivative = applyFunctionOverMatrix(z, ACTIVATION_TYPE[activationType].derivative);
    delta = multiplyMatrix(
      layerDerivative,
      crossProduct(transposeMatrix(weights[i]), delta)
    );
    nablaB[i - 1] = delta;
    nablaW[i - 1] = crossProduct(delta, transposeMatrix(activations[i - 1]));
  }
  return { nablaB, nablaW };
}

export function updateMiniBatch(batch, eta, initBiases, initWeights, costFunction, activationType) {
  // Initialize to zero values
  const nablaB = zeroMatrix(initBiases);
  const nablaW = zeroMatrix(initWeights);
  const biases = initBiases;
  const weights = initWeights;

  // Using a single mini-batch, adjust the gradient for each backprop
  for (let i = 0; i < batch.length; i++) {
    const deltas = backProp(batch[i].x, batch[i].y, biases, weights, costFunction, activationType);
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

export function feedForward(a, biases, weights, activationType) {
  for (let i = 0; i < biases.length; i++) {
    a = applyFunctionOverMatrix(addMatrix(
      crossProduct(weights[i], a),
      biases[i]
    ), ACTIVATION_TYPE[activationType].func);
  }
  return a;
}

export function evaluateTestData(testData, biases, weights, activationType) {

  const testMap = {}
  let count = 0;

  for (let i = 0; i < testData.length; i++) {
    const xResult = _.flattenDeep(feedForward(testData[i].x, biases, weights, activationType));
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


export function printSnapshot(biases, weights, epoch, onStateUpdate, testData, activationType) {
  /* Ensure references are not being held onto here */
  const snapshot = {
    biases: _.cloneDeep(biases),
    weights: _.cloneDeep(weights)
  }
  if (testData) {
    const {testMap, count} = evaluateTestData(testData, biases, weights, activationType);
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
  const {
    trainingData,
    epochs,
    miniBatchSize,
    eta,
    testData,
    onStateUpdate,
    costFunction,
    activationType
  } = options;

  console.log(`Training data points: ${trainingData.length}`)
  printSnapshot(initialBiases, initialWeights, 0, onStateUpdate, testData, activationType)

  for (let i = 0; i < epochs; i++) {
    const miniBatches = splitIntoMiniBatches(_.shuffle(trainingData), miniBatchSize);

    let curWeights = initialWeights;
    let curBiases = initialBiases;
    miniBatches.forEach((batch, index) => {
      const { weights, biases } = updateMiniBatch(batch, eta, curBiases, curWeights, costFunction, activationType);
      curWeights = weights;
      curBiases = biases;
    });

    printSnapshot(curBiases, curWeights, i + 1, onStateUpdate, testData, activationType)
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
        weights[i][j][k] = guassianGenerator() / Math.sqrt(sizes[i]);
      }
    }
  }
  return weights;
}

// Entry point, light option validation.
export function runNeuralNetwork(options) {
  console.log('Starting...');
  if (options.costFunction === 'quadratic' || options.costFunction === 'crossEntropy'
    && options.activationType === 'sigmoid' || options.activationType === 'tanh') {
    const biases = generateBiases(options.sizes);
    const weights = generateWeights(options.sizes);
    runEpochs(options, biases, weights);
  } else {
    throw new Error('Cost function is required');
  }
}
