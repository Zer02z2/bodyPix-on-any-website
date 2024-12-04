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
  width: number
}

const offsets = { x: 0, y: 0 }
const scale = 2
const handPositions: {
  Left: { x: number; y: number }[]
  Right: { x: number; y: number }[]
} = { Left: [], Right: [] }
const leftGloveImage = new Image()
const rightGloveImage = new Image()
leftGloveImage.src = "https://io.zongzechen.com/undnet/files/glove_left.png"
rightGloveImage.src = "https://io.zongzechen.com/undnet/files/glove_right.png"

const arrows: {
  direction: string
  src: string
  target: HTMLElement | undefined
  scroll: () => void
}[] = [
  {
    direction: "up",
    src: "",
    target: undefined,
    scroll: () => {
      window.scrollBy({ top: -50 })
    },
  },
  {
    direction: "down",
    src: "",
    target: undefined,
    scroll: () => {
      window.scrollBy({ top: 50 })
    },
  },
]

const elementData = (element: HTMLElement): ElementData => {
  const rect = element.getBoundingClientRect()
  const increment = 0.02
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
    width: rect.width,
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
let animateList: ElementData[] = []

const init = async () => {
  const canvases = createElements()
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

const createElements = () => {
  let canvases = []
  for (let i = 0; i < 2; i++) {
    const canvas = document.createElement("canvas")
    canvas.id = "body-canvas"
    canvas.style.position = "fixed"
    canvas.style.bottom = "0px"
    canvas.style.scale = `${scale}`
    canvas.style.transform = `translateY(${-110}px)`
    canvas.style.left = `${window.innerWidth / 2 - 640 / 2}px`
    canvas.style.zIndex = "999"
    canvas.width = 640
    canvas.height = 480
    canvas.style.pointerEvents = "none"
    document.body.appendChild(canvas)
    canvases.push(canvas)
  }
  arrows.forEach((arrow) => {
    const button = document.createElement("div")
    button.innerHTML = arrow.direction
    button.style.width = "200px"
    button.style.height = "60px"
    button.style.display = "flex"
    button.style.alignItems = "center"
    button.style.justifyContent = "center"
    button.style.position = "fixed"
    button.style.left = `${window.innerWidth / 2 - 100}px`
    button.style.zIndex = "1000"
    button.style.backgroundColor = "#c9c9c9"
    button.style.border = "1px solid black"
    if (arrow.direction === "up") {
      button.style.top = "0px"
    } else if (arrow.direction === "down") {
      button.style.bottom = "0px"
    }
    button.id = `body-${arrow.direction}-arrow`
    arrow.target = button
    document.body.appendChild(button)
  })
  return canvases
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
  const backgroundColor = { r: 255, g: 255, b: 255, a: 255 }
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

    if (r === 255 && g === 255 && b === 255 && a > 0) {
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
    //const vector2 = hand.keypoints[12]
    const vector3 = hand.keypoints[9]
    ctx.lineWidth = 1
    ctx.strokeStyle = "red"
    const dist = Math.sqrt(
      (vector1.x - vector3.x) ** 2 + (vector1.y - vector3.y) ** 2
    )
    const margin = dist
    circle(ctx, vector3.x, vector3.y, margin)
    const img = hand.handedness == "Left" ? leftGloveImage : rightGloveImage
    ctx.translate(vector3.x, vector3.y)
    const rotationAngle = getRadian(
      { x: vector3.x - vector1.x, y: vector3.y - vector1.y },
      { x: 0, y: 1 }
    )
    ctx.rotate(rotationAngle)
    ctx.drawImage(img, -margin, -margin * 0.7, margin * 2, margin * 2)
    ctx.rotate(-rotationAngle)
    ctx.translate(-vector3.x, -vector3.y)

    let positions = handPositions[hand.handedness]
    positions.push(vector3)
    const limit = 20
    if (positions.length > limit) positions.slice(0, 1)
    const direction = {
      x: positions[positions.length - 1].x - positions[0].x,
      y: positions[positions.length - 1].y - positions[0].y,
    }

    let detectArrow = false
    arrows.forEach((arrow) => {
      const rect = arrow.target?.getBoundingClientRect()
      if (!rect) return
      const handX = vector3.x * 2 + offsets.x
      const handY = vector3.y * 2 + offsets.y
      if (
        handX > rect.left - margin &&
        handX < rect.left + rect.width + margin &&
        handY > rect.top - margin &&
        handY < rect.top + rect.height + margin
      ) {
        arrow.scroll()
        detectArrow = true
      }
    })
    if (detectArrow) return

    const overlappedElement = findOverlapElement(
      vector3.x,
      vector3.y,
      margin,
      endChildren
    )
    if (
      !overlappedElement ||
      animateList.find(
        (element) => element.target === overlappedElement.target
      ) ||
      overlappedElement.target == document.body ||
      overlappedElement.target.id === "body-up-arrow" ||
      overlappedElement.target.id === "body-down-arrow"
    )
      return
    const children = Array.from(
      overlappedElement.target.childNodes
    ) as HTMLElement[]
    let childrenInAnimation = false
    children.forEach((child) => {
      if (animateList.find((element) => element.target === child)) {
        childrenInAnimation = true
      }
    })
    if (childrenInAnimation) return

    overlappedElement.moveX = direction.x * 20
    overlappedElement.moveY = direction.y * 10
    animateList.push(overlappedElement)
  })

  window.requestAnimationFrame(() => {
    analyzeHand(detector, video, canvas)
  })
}

const findOverlapElement = (
  x: number,
  y: number,
  margin: number,
  elements: ElementData[]
) => {
  const handX = x * 2 + offsets.x + window.scrollX
  const handY = y * 2 + offsets.y + document.body.scrollTop
  const result = elements.find((element) => {
    const { xLeft, xRight, yTop, yBottom, target } = element
    if (
      handX > xLeft - margin &&
      handX < xRight + margin &&
      handY > yTop - margin &&
      handY < yBottom + margin &&
      target.id !== "body-canvas"
    ) {
      return true
    } else {
      return false
    }
  })
  if (result) return result

  const parentList = elements
    .map((element) => {
      const parent = element.target.parentElement
      if (parent) return elementData(parent)
      else return null
    })
    .filter((value, index, array) => {
      return array.indexOf(value) === index
    })
    .filter((element) => element !== null)

  if (parentList.length === 0) return null
  else return findOverlapElement(x, y, margin, parentList)
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
      endChildren = findEndChildren()
    }
  }
  window.requestAnimationFrame(animate)
}
animate()

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

const getRadian = (
  vecA: { x: number; y: number },
  vecB: { x: number; y: number }
) => {
  const crossProduct = vecA.x * vecB.y - vecA.y * vecB.x
  const magnitudeA = Math.sqrt(vecA.x ** 2 + vecA.y ** 2)
  const magnitudeB = Math.sqrt(vecB.x ** 2 + vecB.y ** 2)
  const sineTheta = crossProduct / (magnitudeA * magnitudeB)
  const clampedSineTheta = Math.max(-1, Math.min(1, sineTheta))
  let result = Math.asin(clampedSineTheta)
  if (vecA.y > 0) {
    if (vecA.x < 0) {
      result = -Math.PI - result
    } else result = Math.PI - result
  }
  return result
}
