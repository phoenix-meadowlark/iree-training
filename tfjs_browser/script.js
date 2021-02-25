// Experiment Parameters
const TFJS_BACKEND = "wasm";
const MULTI_THREAD = true;

const BENCHMARK_ITERATIONS = 1000;
const MS_BETWEEN_ITERATIONS = 0;

const BENCHMARK_PREDICT = false;
const BENCHMARK_FIT = false;
const BENCHMARK_TRAIN_ON_BATCH = true;

const MODEL_NAME = "mnist_dnn";
const BATCH_SIZE = 32;
const OPTIMIZER = "sgd";
const LEARNING_RATE = 0.01; // Should have not effect.
const OPTIMIZERS = {
  sgd: tf.train.sgd(LEARNING_RATE),
  adadelta: tf.train.adadelta(LEARNING_RATE),
  adagrad: tf.train.adagrad(LEARNING_RATE),
  adam: tf.train.adam(LEARNING_RATE),
  adamax: tf.train.adamax(LEARNING_RATE),
  momentum: tf.train.momentum(LEARNING_RATE, 0.1),
  rmsprop: tf.train.rmsprop(LEARNING_RATE),
}

console.log("model_name:            " + MODEL_NAME);
console.log("batch_size:            " + BATCH_SIZE);
console.log("optimizer:             " + OPTIMIZER);
console.log("benchmark_iterations:  " + BENCHMARK_ITERATIONS);
console.log("ms_between_iterations: " + MS_BETWEEN_ITERATIONS);


function get_compilation_options() {
  return {
    optimizer: OPTIMIZERS[OPTIMIZER],
    loss: "categoricalCrossentropy",
  };
};


function sleep(ms) {
  // https://stackoverflow.com/a/39914235
  return new Promise(resolve => setTimeout(resolve, ms));
}


async function timeModelPredict(model, images) {
  // don't time the jit compilation.
  console.log("compiling model.predict");
  const warmup = await model.predict(images);
  warmup.dispose();

  console.log("benchmarking model.predict");
  let times = [];
  for (let i = 0; i < BENCHMARK_ITERATIONS; i++) {
    let startTime = window.performance.now();
    results = await model.predict(images);
    await results.data();
    results.dispose();
    let endTime = window.performance.now();
    times.push((endTime - startTime) / 1000.0);

    const status = document.getElementById("modelPredictStatus");
    status.innerText = "model.predict times:\n" + JSON.stringify(times) + ",";
    await sleep(MS_BETWEEN_ITERATIONS);
  }
  console.log(JSON.stringify(times) + ",");
  await sleep(1000);
}


async function getHistoryResults(history) {
  // Ensure that all of the values of h.history are set before finishing
  // the benchmark iteration.
  const promises = [];
  const tensors = [];
  for (const key in history.history) {
    const valueArray = history.history[key];
    for (let i = 0; i < valueArray.length; ++i) {
      if (typeof valueArray[i] !== "number") {
        const valueScalar = valueArray[i];
        tensors.push(valueScalar);
        promises.push(valueScalar.data());
      }
    }
  }
  const values = await Promise.all(promises);
  return tensors
}


async function timeModelFit(model, images, labels) {
  // don't time the jit compilation.
  console.log("compiling model.fit");
  const history = await model.fit(images, labels,
                                  { epochs: 1, batchSize: BATCH_SIZE });
  const tensors = await getHistoryResults(history);
  tf.dispose(tensors);

  console.log("benchmarking model.fit");
  let times = [];
  for (let i = 0; i < BENCHMARK_ITERATIONS; i++) {
    let startTime = window.performance.now();
    const history = await model.fit(images, labels,
                                    { epochs: 1, batchSize: BATCH_SIZE });
    const tensors = await getHistoryResults(history);
    tf.dispose(tensors);
    let endTime = window.performance.now();
    times.push((endTime - startTime) / 1000.0);

    const status = document.getElementById("modelFitStatus");
    status.innerText = "model.fit times:\n" + JSON.stringify(times) + ",";
    await sleep(MS_BETWEEN_ITERATIONS);
  }
  // Log all times in a copy-pastable form if training completes.
  console.log(JSON.stringify(times) + ",");
  await sleep(1000);
}


async function timeModelTrainOnBatch(model, images, labels) {
  // don't time the jit compilation.
  console.log("compiling model.trainOnBatch");
  const history = await model.trainOnBatch(images, labels);
  const tensors = await getHistoryResults(history);
  tf.dispose(tensors);

  console.log("benchmarking model.trainOnBatch");
  let times = [];
  for (let i = 0; i < BENCHMARK_ITERATIONS; i++) {
    let startTime = window.performance.now();
    const history = await model.trainOnBatch(images, labels);
    const tensors = await getHistoryResults(history);
    let endTime = window.performance.now();
    tf.dispose(tensors);
    times.push((endTime - startTime) / 1000.0);

    const status = document.getElementById("modelTrainOnBatchStatus");
    status.innerText = `model.trainOnBatch ${i+1}/${BENCHMARK_ITERATIONS}:\n${JSON.stringify(times)},`;
    await sleep(MS_BETWEEN_ITERATIONS);
  }
  // Log all times in a copy-pastable form if training completes.
  console.log(JSON.stringify(times) + ",");
  await sleep(1000);
}


async function benchmarkModel(model, images, labels, model_name) {
  if (BENCHMARK_PREDICT) {
    console.log("Benchmarking " + model_name + ".predict");
    await timeModelPredict(model, images);
  }

  if (BENCHMARK_FIT) {
    console.log("Benhcmarking " + model_name + ".fit");
    await timeModelFit(model, images, labels);
  }

  if (BENCHMARK_TRAIN_ON_BATCH) {
    console.log("Benhcmarking " + model_name + ".trainOnBatch");
    await timeModelTrainOnBatch(model, images, labels);
  }
}


async function benchmarkMnistDnn() {
  const image_size = [28, 28, 1];
  const num_classes = 10;
  const images = tf.randomNormal([BATCH_SIZE, ...image_size]);
  const labels = tf.oneHot(Array(BATCH_SIZE).fill(0), num_classes);

  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: image_size }));
  model.add(tf.layers.dense({ units: 128 }));
  model.add(tf.layers.reLU());
  model.add(tf.layers.dense({ units: num_classes }));
  model.add(tf.layers.softmax());
  model.compile(get_compilation_options());

  benchmarkModel(model, images, labels, "mnist_dnn");
}


async function benchmarkMobileNet() {
  const image_size = [224, 224, 3];
  const num_classes = 1000;
  const images = tf.randomNormal([BATCH_SIZE, ...image_size]);
  const labels = tf.oneHot(Array(BATCH_SIZE).fill(0), num_classes);

  console.log("Loading MobileNetV1");
  const mobilenet = await tf.loadLayersModel(
    "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json");
  mobilenet.compile(get_compilation_options());

  benchmarkModel(model, images, labels, "mobilenet");
}

tf.env().set('WEBGL_FLUSH_THRESHOLD', 1);
tf.env().set('WASM_HAS_MULTITHREAD_SUPPORT', MULTI_THREAD)
tf.setBackend(TFJS_BACKEND).then(async () => {
  console.log("TF.js backend:         " + tf.getBackend());
  console.log("TF.js version:         " + tf.version.tfjs);

  if (MODEL_NAME == "mnist_dnn") {
    benchmarkMnistDnn();
  } else if (MODEL_NAME == "mobilenet") {
    benchmarkMobileNet();
  } else {
    throw "MODEL_NAME must be 'mnist_dnn' or 'mobilent', not " + MODEL_NAME;
  }
});
