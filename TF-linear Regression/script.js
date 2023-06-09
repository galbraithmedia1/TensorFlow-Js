const status = document.getElementById("status");
if (status) {
  status.innerText = "Loaded TensorFlow.js - version: " + tf.version.tfjs;
}

console.log("hello world");

import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js";

const INPUTS = TRAINING_DATA.inputs;

const OUTPUTS = TRAINING_DATA.outputs;

tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUTS_TENSOR = tf.tensor2d(INPUTS);

const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS);

function normalize(tensor, min, max) {
  const result = tf.tidy(function () {
    const MIN_VALUES = min || tf.min(tensor, 0);

    const MAX_VALUES = max || tf.max(tensor, 0);

    // now sutrac the MIN_VALUES fromevery value in the tensor and store the resules in a new tensor
    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);

    // calculate the range size of the possile values
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

    // canculate the adjusted values dividd y the range size as a new tensor
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

    return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES };
  });

  return result;
}

//normalize the input features arrays and then dispose of the original non normalized tensors

const FEATURE_RESULTS = normalize(INPUTS_TENSOR);
console.log(" normailzed values");

FEATURE_RESULTS.NORMALIZED_VALUES.print();

console.log(" min values");
FEATURE_RESULTS.MIN_VALUES.print();

console.log(" max values");
FEATURE_RESULTS.MAX_VALUES.print();

INPUTS_TENSOR.dispose();

const model = tf.sequential();

model.add(tf.layers.dense({ inputShape: [2], units: 1 }));

model.summary();

train();

async function train() {
  const LEARNING_RATE = 0.01; // Choose learning rate

  // compile the model with the defined learnign rate and specify a loss function to use

  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: "meanSquaredError",
  });

  // train itself
  let results = await model.fit(
    FEATURE_RESULTS.NORMALIZED_VALUES,
    OUTPUTS_TENSOR,
    {
      validationSplit: 0.15, // 15% of the data will be used for validation
      shuffle: true, // shuffle the data before each epoch
      batchSize: 64, // specify the batch size
      epochs: 10, // specify the number of epochs
    }
  );

  OUTPUTS_TENSOR.dispose();
  FEATURE_RESULTS.NORMALIZED_VALUES.dispose();

  console.log(
    " average error loss " +
      Math.sqrt(results.history.loss[results.history.loss.length - 1])
  );
  console.log(
    " average validation loss " +
      Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1])
  );

  evaluate();
}

function evaluate() {
  tf.tidy(function () {
    let newInput = normalize(
      tf.tensor2d([[750, 1]]),
      FEATURE_RESULTS.MIN_VALUES,
      FEATURE_RESULTS.MAX_VALUES
    );
    let output = model.predict(newInput.NORMALIZED_VALUES);
    output.print();
  });

  FEATURE_RESULTS.MIN_VALUES.dispose();
  FEATURE_RESULTS.MAX_VALUES.dispose();
  model.dispose();

  console.log(tf.memory().numTensors);
}
