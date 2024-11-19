import * as bodySegmentation from '@tensorflow-models/body-segmentation'
import '@tensorflow/tfjs-core'
import '@tensorflow/tfjs-backend-webgl'
import '@mediapipe/selfie_segmentation'

const model = bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation
const segmenterConfig = {
  runtime: 'mediapipe',
  solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation'
};

