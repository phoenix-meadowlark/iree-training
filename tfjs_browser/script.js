function sleep(ms) {
  // https://stackoverflow.com/a/39914235
  return new Promise(resolve => setTimeout(resolve, ms));
}

tf.setBackend('cpu').then(async () => {
  console.log(tf.getBackend());

  const BENCHMARK_ITERATIONS = 10;

  const EPOCHS = 1;
  const BATCH_SIZE = 32;
  const IMAGE_SIZE = [224, 224, 3]
  const NUM_CLASSES = 1000

  const IMAGES = tf.randomNormal([BATCH_SIZE, ...IMAGE_SIZE]);
  const LABELS = tf.oneHot(Array(BATCH_SIZE).fill(0), NUM_CLASSES);

  function get_compilation_options() {
    return {
      optimizer: tf.train.sgd(0.01),
      loss: 'categoricalCrossentropy'
    };
  };


  async function timeModelPredict(model) {
    // don't time the jit compilation.
    console.log("compiling model.predict");
    await model.predict(IMAGES);

    console.log("benchmarking model.predict");
    let times = [];
    for (let i = 0; i < BENCHMARK_ITERATIONS; i++) {
      let startTime = window.performance.now();
      await model.predict(IMAGES);
      let endTime = window.performance.now();
      times.push(endTime - startTime);
    }
    console.log(JSON.stringify(times));
    await sleep(1000);
  }


  async function timeModelFit(model) {
    // don't time the jit compilation.
    console.log("compiling model.fit");
    await model.fit(IMAGES, LABELS, { epochs: EPOCHS, batchSize: BATCH_SIZE });

    console.log("benchmarking model.fit");
    let times = [];
    for (let i = 0; i < BENCHMARK_ITERATIONS; i++) {
      let startTime = window.performance.now();
      await model.fit(IMAGES, LABELS, { epochs: EPOCHS, batchSize: BATCH_SIZE });
      let endTime = window.performance.now();
      times.push(endTime - startTime);
    }
    console.log(JSON.stringify(times));
    await sleep(1000);
  }


  async function trainSimpleDnn() {
    console.log("compiling simple dnn");
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: IMAGE_SIZE }));
    model.add(tf.layers.dense({ units: 128 }));
    model.add(tf.layers.reLU());
    model.add(tf.layers.dense({ units: NUM_CLASSES }));
    model.add(tf.layers.softmax());
    model.compile(get_compilation_options());

    console.log("benchmarking simple_dnn.predict");
    await timeModelPredict(model);

    console.log("benhcmarking simple_dnn.fit");
    await timeModelFit(model);
  }

  await trainSimpleDnn();

  async function trainMobileNet() {
    console.log("loading mobilenet");
    const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    mobilenet.compile(get_compilation_options());
    // mobilenet.summary();

    console.log("benchmarking mobilenet.predict");
    await timeModelPredict(mobilenet);

    console.log("benchmarking mobilenet.fit");
    await timeModelFit(mobilenet);
  }

  // await trainMobileNet();

  console.log("finished.");

});
