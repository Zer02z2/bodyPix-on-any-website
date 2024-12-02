import * as bodySegmentation from "@tensorflow-models/body-segmentation"
import * as handPoseDetection from "@tensorflow-models/hand-pose-detection"
import "@tensorflow/tfjs-core"
import "@tensorflow/tfjs-backend-webgl"
import "@mediapipe/selfie_segmentation"
import "@mediapipe/hands"

interface ElementData {
  target: HTMLElement
  moveX: number
  moveY: number
  increment: number
  progress: number
  opacity: number
  scale: number
  xLeft: number
  xRight: number
  yTop: number
  yBottom: number
}

const offsets = { x: 0, y: 0 }
const scale = 2
const handPositions: {
  Left: { x: number; y: number }[]
  Right: { x: number; y: number }[]
} = { Left: [], Right: [] }

const elementData = (element: HTMLElement): ElementData => {
  const rect = element.getBoundingClientRect()
  const increment = 0.04
  return {
    target: element,
    moveX: 0,
    moveY: 0,
    increment: increment,
    progress: 0,
    opacity: 1,
    scale: 1,
    xLeft: rect.left,
    xRight: rect.left + rect.width,
    yTop: rect.top,
    yBottom: rect.top + rect.height,
  }
}

const findEndChildren = () => {
  const elementsNodes = document.body.querySelectorAll("*")
  const elements = Array.from(elementsNodes)
  let result: ElementData[] = []
  for (let i = elements.length - 1; i >= 0; i--) {
    const element = elements[i] as HTMLElement
    const children = element.querySelectorAll(":scope > *")
    if (Array.from(children).length === 0) {
      result.push(elementData(element))
    }
  }
  return result
}

let endChildren = findEndChildren()
let watchList: ParentNode[] = []
let animateList: ElementData[] = []

const init = async () => {
  let canvases = []
  for (let i = 0; i < 2; i++) {
    const canvas = document.createElement("canvas")
    canvas.style.position = "fixed"
    canvas.style.bottom = "0px"
    canvas.style.scale = `${scale}`
    canvas.style.transform = `translateY(${-110}px)`
    canvas.style.left = `${window.innerWidth / 2 - 640 / 2}px`
    canvas.style.zIndex = "99"
    canvas.width = 640
    canvas.height = 480
    canvas.style.pointerEvents = "none"
    document.body.appendChild(canvas)
    canvases.push(canvas)
  }
  const video = await initWebcam()
  video.addEventListener("loadeddata", async () => {
    const bodySegmenter = await initBodySeg()
    const handDetector = await initHandPose()
    const rect = canvases[0].getBoundingClientRect()
    offsets.x = rect.left
    offsets.y = rect.top
    analyzeBody(bodySegmenter, video, canvases[0])
    analyzeHand(handDetector, video, canvases[1])
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
  const model = bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation
  const segmenterConfig: bodySegmentation.MediaPipeSelfieSegmentationMediaPipeModelConfig =
    {
      runtime: "mediapipe",
      solutionPath:
        "https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation",
    }
  const segmenter = await bodySegmentation.createSegmenter(
    model,
    segmenterConfig
  )
  return segmenter
}

const initHandPose = async () => {
  const model = handPoseDetection.SupportedModels.MediaPipeHands
  const detectorConfig: handPoseDetection.MediaPipeHandsMediaPipeModelConfig = {
    runtime: "mediapipe",
    solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/hands",
    // or 'base/node_modules/@mediapipe/hands' in npm.
  }
  const detector = await handPoseDetection.createDetector(model, detectorConfig)
  return detector
}

const analyzeBody = async (
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
    analyzeBody(segmenter, video, canvas)
  })
}

const analyzeHand = async (
  detector: handPoseDetection.HandDetector,
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement
) => {
  const estimationConfig = { flipHorizontal: true }
  const hands = await detector.estimateHands(video, estimationConfig)

  const ctx = canvas.getContext("2d")
  if (!ctx) return

  ctx.clearRect(0, 0, canvas.width, canvas.height)

  hands.forEach((hand) => {
    const vector1 = hand.keypoints[0]
    const vector2 = hand.keypoints[12]
    const vector3 = hand.keypoints[9]
    ctx.lineWidth = 1
    ctx.strokeStyle = "red"
    //line(ctx, vector1.x, vector1.y, vector2.x, vector2.y)
    const dist = Math.sqrt(
      (vector1.x - vector2.x) ** 2 + (vector1.y - vector2.y) ** 2
    )
    const margin = dist * 0.5
    circle(ctx, vector3.x, vector3.y, margin)

    let positions = handPositions[hand.handedness]
    positions.push(vector3)
    const limit = 20
    if (positions.length > limit) positions.slice(0, 1)
    const direction = {
      x: positions[positions.length - 1].x - positions[0].x,
      y: positions[positions.length - 1].y - positions[0].y,
    }

    for (let i = endChildren.length - 1; i >= 0; i--) {
      let element = endChildren[i]
      if (!animateList.find((data) => data.target === element.target)) {
        element = elementData(element.target)
      }
      const handX = vector3.x * 2 + offsets.x + window.scrollX
      const handY = vector3.y * 2 + offsets.y + window.scrollY
      console.log(handY)
      const { xLeft, xRight, yTop, yBottom } = element
      if (
        handX > xLeft - margin &&
        handX < xRight + margin &&
        handY > yTop - margin &&
        handY < yBottom + margin
      ) {
        element.moveX = direction.x * 10
        element.moveY = direction.y * 10
        animateList.push(element)
        endChildren.splice(i, 1)
      }
    }
  })

  window.requestAnimationFrame(() => {
    analyzeHand(detector, video, canvas)
  })
}

init()
const animate = () => {
  for (let i = animateList.length - 1; i >= 0; i--) {
    const element = animateList[i]
    const { target, moveX, moveY, increment, progress, opacity, scale } =
      element

    element.progress = progress + increment
    element.opacity = opacity - increment
    element.scale = scale - increment
    target.style.transform = `translateX(${moveX * progress}px) translateY(${
      moveY * progress
    }px)`
    target.style.opacity = `${opacity}`
    target.style.scale = `${scale}`

    if (element.progress >= 1) {
      animateList.splice(i, 1)
      const parent = target.parentNode
      if (!parent) break
      parent.removeChild(target)
      if (watchList.find((item) => item === parent)) break
      watchList.push(parent)
    }
    for (let i = watchList.length - 1; i >= 0; i--) {
      const element = watchList[i] as HTMLElement
      if (element.querySelectorAll(":scope > *").length === 0) {
        watchList.splice(i, 1)
        endChildren.push(elementData(element))
      }
    }
  }
  window.requestAnimationFrame(animate)
}
animate()

// const line = (
//   ctx: CanvasRenderingContext2D,
//   x1: number,
//   y1: number,
//   x2: number,
//   y2: number
// ) => {
//   ctx.beginPath()
//   ctx.moveTo(x1, y1)
//   ctx.lineTo(x2, y2)
//   ctx.stroke()
// }

const circle = (
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number
) => {
  ctx.beginPath()
  ctx.arc(x, y, radius, 0, 2 * Math.PI)
  ctx.stroke()
}
