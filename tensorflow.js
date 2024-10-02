/*
Javascript Translation of Python Tensorflow tutorial made by Morvan Zhou
Original Python code: https://github.com/MorvanZhou/Tensorflow-Tutorial/blob/master/tutorial-contents/406_GAN.py

Dependencies:
tensorflow.js
chart.js

Translation made by Gustav Holmström
*/

const EPOCHS = 5000;
const BATCH_SIZE = 64;
const LR_G = 0.0001;  // Learning rate for the generator
const LR_D = 0.0001;  // Learning rate for the discriminator
const N_IDEAS = 5;    // Number of random ideas for generating art
const ART_COMPONENTS = 15;  // Number of points G can draw
const PAINT_POINTS = tf.linspace(-1, 1, ART_COMPONENTS).reshape([1, ART_COMPONENTS]).tile([BATCH_SIZE, 1]); // Generate painting points (between -1 and 1)

// Generator model
const generator = tf.sequential();
generator.add(tf.layers.dense({ units: 128, inputShape: [N_IDEAS], activation: 'relu' }));
generator.add(tf.layers.dense({ units: ART_COMPONENTS }));

// Discriminator model
const discriminator = tf.sequential();
discriminator.add(tf.layers.dense({ units: 128, inputShape: [ART_COMPONENTS], activation: 'relu' }));
discriminator.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

const optimizerD = tf.train.adam(LR_D);
const optimizerG = tf.train.adam(LR_G);

const chart = createChart()
train()

function generateRealArtworks() {
  const a = tf.randomUniform([BATCH_SIZE, 1], 1, 2);
  const paintings = a.mul(PAINT_POINTS.square()).add(a.sub(1));
  return paintings;
}

function gLoss(probabilityFake) {
  return tf.mean(tf.log(tf.sub(1, probabilityFake)));
}

function dLoss(probabilityFake, probabilityReal) {
  return tf.neg(tf.mean(tf.add(tf.log(probabilityReal), tf.log(tf.sub(1, probabilityFake)))));
}


async function train() {
  
  for (let step = 0; step < EPOCHS; step++) {
    await tf.tidy(() => { // tensor memory inside this scope will be cleaned
      const realPaintings = generateRealArtworks();
      const ideas = tf.randomNormal([BATCH_SIZE, N_IDEAS]);

      // Update the Discriminator
      discriminator.trainable = true
      const generatedPpaintings = generator.predict(ideas); // predict this outside minimize
      dResult = optimizerD.minimize(() => {
        const probabilityReal = discriminator.predict(realPaintings)
        const probabilityFake = discriminator.predict(generatedPpaintings)
        return dLoss(probabilityFake, probabilityReal)
      }, true, discriminator.trainableVariables);
      discriminator.trainable = false // dont train discriminator during generation

      // Update the Generator
      gResult = optimizerG.minimize(() => {
        const probabilityFake = discriminator.predict(generator.predict(ideas));
        return gLoss(probabilityFake)
      }, true, generator.trainableVariables);

      // update accuracy and score display
      document.getElementById('dScoreText').innerText = `D score = ${-dResult.dataSync()[0].toFixed(2)} (-1.38 for G to converge)`;
      const dPredicts = discriminator.predict(realPaintings).dataSync();
      const dAccuracy = dPredicts.reduce((a, b) => a + b, 0) / dPredicts.length;
      document.getElementById('dAccuracyText').innerText = `D accuracy = ${dAccuracy.toFixed(2)} (0.5 for D to converge)`;

      // show example of generated art
      chart.data.datasets[0].data = generatedPpaintings.arraySync()[0];
    })

    // Update chart at set speed
    if (step % 50 === 0) {
      await new Promise(resolve => setTimeout(resolve, 0))
      chart.update()
    }
  }
}

function createChart() {
  const ctx = document.getElementById('chart').getContext('2d');
  const chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: Array.from({ length: ART_COMPONENTS }, (_, i) => -1 + (2 * i) / (ART_COMPONENTS - 1)),
      datasets: [
        {
          label: 'Generated Painting',
          data: [],
          borderColor: '#4AD631',
          fill: false,
          tension: 0.1
        },
        {
          label: 'Upper Bound',
          data: [],
          borderColor: '#74BCFF',
          fill: false,
          borderDash: [5, 5],
          tension: 0.1
        },
        {
          label: 'Lower Bound',
          data: [],
          borderColor: '#FF9359',
          fill: false,
          borderDash: [5, 5],
          tension: 0.1
        }
      ]
    },
    options: {
      responsive: true,
      animation: false,
      scales: {
        x: {
          type: 'linear',
          min: -1,
          max: 1,
          ticks: {
            callback: function (value) {
              return value.toFixed(1);  // Format labels to 1 decimal places
            }
          }
        },
        y: {
          min: 0,
          max: 3
        }
      }
    }
  });
  chart.data.datasets[1].data = PAINT_POINTS.square().mul(2).add(1).arraySync()[0]; //upper bound
  chart.data.datasets[2].data = PAINT_POINTS.square().arraySync()[0]; // lower bound
  return chart
}