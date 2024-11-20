import * as bodySegmentation from "@tensorflow-models/body-segmentation"
import "@tensorflow/tfjs-core"
import "@tensorflow/tfjs-backend-webgl"
import "@mediapipe/selfie_segmentation"

const init = async () => {
  const canvas = document.createElement("canvas")
  canvas.style.position = "fixed"
  canvas.style.bottom = "0px"
  canvas.style.scale = "2"
  canvas.style.transform = "translateY(-110px)"
  canvas.style.left = `${window.innerWidth / 2 - 640 / 2}px`
  canvas.style.zIndex = "99"
  canvas.width = 640
  canvas.height = 480
  canvas.style.pointerEvents = "none"
  document.body.appendChild(canvas)
  const video = await initWebcam()
  video.addEventListener("loadeddata", async () => {
    const segmenter = await initBodySeg()
    await analyze(segmenter, video, canvas)
  })
}

const initWebcam = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true })
  const video = document.createElement("video")
  video.srcObject = stream
  video.play()
  video.width = 640
  video.height = 480
  return video
}

const initBodySeg = async () => {
  console.log(1)
  const model = bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation
  console.log(2)
  const segmenterConfig: bodySegmentation.MediaPipeSelfieSegmentationMediaPipeModelConfig =
    {
      runtime: "mediapipe",
      solutionPath:
        "https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation",
    }
  console.log(3)
  console.log(model)
  const segmenter = await bodySegmentation.createSegmenter(
    model,
    segmenterConfig
  )
  console.log(4)
  return segmenter
}

const analyze = async (
  segmenter: bodySegmentation.BodySegmenter,
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement
) => {
  const segmentationConfig = { flipHorizontal: false }
  const people = await segmenter.segmentPeople(video, segmentationConfig)

  const foregroundColor = { r: 0, g: 0, b: 0, a: 0 }
  const backgroundColor = { r: 0, g: 0, b: 0, a: 255 }
  const backgroundDarkeningMask = await bodySegmentation.toBinaryMask(
    people,
    foregroundColor,
    backgroundColor
  )

  const opacity = 1
  const maskBlurAmount = 3
  const flipHorizontal = true
  await bodySegmentation.drawMask(
    canvas,
    video,
    backgroundDarkeningMask,
    opacity,
    maskBlurAmount,
    flipHorizontal
  )

  const ctx = canvas.getContext("2d")
  if (!ctx) return
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
  const data = imageData.data

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i]
    const g = data[i + 1]
    const b = data[i + 2]
    const a = data[i + 3]

    if (r === 0 && g === 0 && b === 0 && a > 0) {
      data[i + 3] = 0
    }
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.putImageData(imageData, 0, 0)
  ctx.clearRect(0, 0, canvas.width * 0.02, canvas.height)
  ctx.clearRect(canvas.width * 0.98, 0, canvas.width * 0.02, canvas.height)
  ctx.clearRect(0, 0, canvas.width, canvas.height * 0.02)
  ctx.clearRect(0, canvas.height * 0.98, canvas.width, canvas.height * 0.02)

  window.requestAnimationFrame(() => {
    analyze(segmenter, video, canvas)
  })
}

init()
