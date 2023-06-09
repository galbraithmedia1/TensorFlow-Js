const status = document.getElementById("status");
if (status) {
  status.innerText = "Loaded TensorFlow.js - version: " + tf.version.tfjs;
}

import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js";

// gra a reference to the canvas element
const INPUTS = TRAINING_DATA.inputs;
// grab reference to the minst data
const OUTPUTS = TRAINING_DATA.outputs;

// shuffle the data

tf.util.shuffleCombo(INPUTS, OUTPUTS);

// convert the data to tensors
const INPUTS_TENSOR = tf.tensor2d(INPUTS);

// convert the data to tensors

const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, "int32"), 10);

const model = tf.sequential();

model.add(
  tf.layers.dense({ inputShape: [784], units: 32, activation: "relu" })
);
model.add(tf.layers.dense({ units: 16, activation: "relu" }));
model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

model.summary();

train();

// define a function to log the progress of the training
function logProgress(epoch, logs) {
  console.log(" epoch " + epoch + " loss " + Math.sqrt(logs.loss));

  if (epoch == 70) {
    OPTIMIZER.setLearningRate(LEARNING_RATE / 2);
  }
}

async function train() {
  // compile the model with the defined learnign rate and specify a loss function to use

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // train itself
  let results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
    callbacks: { onEpochEnd: logProgress },

    validationSplit: 0.2, // 15% of the data will be used for validation
    shuffle: true, // shuffle the data before each epoch
    batchSize: 512, // specify the batch size
    epochs: 50, // specify the number of epochs
  });

  OUTPUTS_TENSOR.dispose();
  INPUTS_TENSOR.dispose();

  evaluate();
}

const PREDICTION_ELEMENT = document.getElementById("prediction");

function evaluate() {
  const OFFSET = Math.floor(Math.random() * INPUTS.length);

  let answer = tf.tidy(function () {
    let newInput = tf.tensor1d(INPUTS[OFFSET]);

    let output = model.predict(newInput.expandDims());
    output.print();
    return output.squeeze().argMax();
  });

  answer.array().then(function (index) {
    PREDICTION_ELEMENT.innerText = "Prediction: " + index;
    PREDICTION_ELEMENT.setAttribute(
      "class",
      index === OUTPUTS[OFFSET] ? "correct" : "wrong"
    );
    answer.dispose();
    drawImage(INPUTS[OFFSET]);
  });
}

const CANVAS = document.getElementById("canvas");
const CTX = CANVAS.getContext("2d");

function drawImage(digit) {
  var imageData = CTX.getImageData(0, 0, 28, 28);

  for (let i = 0; i < digit.length; i++) {
    imageData.data[i * 4] = digit[i] * 255; // R
    imageData.data[i * 4 + 1] = digit[i] * 255; // G
    imageData.data[i * 4 + 2] = digit[i] * 255; // B
    imageData.data[i * 4 + 3] = 255; // A
  }

  CTX.putImageData(imageData, 0, 0);
  setTimeout(evaluate, 800);
}
